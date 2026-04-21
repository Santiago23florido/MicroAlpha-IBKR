from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from config import Settings
from data.lob_dataset import load_lob_capture_frame
from models.deep_lob import DeepLOBReferenceLike, lob_feature_columns
from models.registry import ModelRegistry


CLASS_LABELS = {0: "down", 1: "stationary", 2: "up"}
BTC_MIN_ORDER_FALLBACK = 0.00005


@dataclass(frozen=True)
class KrakenMarketMinimum:
    symbol: str
    pair_key: str
    minimum_order_base: float
    price_eur: float
    source: str
    warning: str | None = None

    @property
    def minimum_order_notional_eur(self) -> float:
        return float(self.minimum_order_base) * float(self.price_eur)


@dataclass(frozen=True)
class KrakenPaperPolicy:
    initial_cash_eur: float
    initial_cash_mode: str
    initial_cash_source: str
    minimum_order_base: float
    minimum_order_notional_eur: float
    minimum_order_source: str
    position_fraction: float
    fee_bps: float
    slippage_bps: float
    model_prob_threshold: float
    max_trades_per_day: int
    max_daily_loss_pct: float
    max_open_positions: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KrakenPaperState:
    cash_eur: float
    position_qty: float = 0.0
    total_fees_eur: float = 0.0
    realized_pnl_eur: float = 0.0
    entry_notional_eur: float = 0.0


class KrakenPaperSimulator:
    def __init__(
        self,
        *,
        settings: Settings,
        symbol: str,
        artifact_path: Path,
        artifact_payload: dict[str, Any],
        policy: KrakenPaperPolicy,
    ) -> None:
        self.settings = settings
        self.symbol = symbol.upper()
        self.artifact_path = artifact_path
        self.artifact_payload = artifact_payload
        self.policy = policy
        self.depth_levels = int(artifact_payload.get("depth_levels", settings.models.lob_depth_levels))
        self.sequence_length = int(artifact_payload.get("sequence_length", settings.models.lob_sequence_length))
        self.feature_columns = lob_feature_columns(self.depth_levels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepLOBReferenceLike(
            feature_dim=self.depth_levels * 4,
            sequence_length=self.sequence_length,
        )
        self.model.load_state_dict(artifact_payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.state = KrakenPaperState(cash_eur=float(policy.initial_cash_eur))
        self.trades: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.equity_curve: list[dict[str, Any]] = []

    def run(self, replay: pd.DataFrame) -> None:
        if len(replay) < self.sequence_length:
            raise ValueError(
                f"Not enough rows for paper simulation. Need at least {self.sequence_length}, got {len(replay)}."
            )
        for end_index in range(self.sequence_length - 1, len(replay)):
            window = replay.iloc[end_index - self.sequence_length + 1 : end_index + 1]
            self.step(window)

    def step(self, window: pd.DataFrame) -> dict[str, Any]:
        row = window.iloc[-1]
        features = _normalize_window(window, self.feature_columns, self.depth_levels)
        with torch.no_grad():
            logits, _predicted_return = self.model(torch.tensor(features[None, :, :], dtype=torch.float32, device=self.device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        timestamp = str(row["event_ts_utc"])
        bid = float(row["bid_px_1"])
        ask = float(row["ask_px_1"])
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        action = "hold"
        reason = "stationary_or_no_position_signal"
        execution_price = 0.0
        quantity = 0.0
        fee = 0.0

        if bid <= 0 or ask <= 0:
            reason = "invalid_top_of_book"
        elif confidence < self.policy.model_prob_threshold:
            reason = "confidence_below_threshold"
        elif self._daily_loss_limit_reached(bid):
            reason = "daily_loss_limit_reached"
        elif predicted_class == 2:
            action, reason, execution_price, quantity, fee = self._try_buy(timestamp, ask, confidence)
        elif predicted_class == 0:
            action, reason, execution_price, quantity, fee = self._try_sell(timestamp, bid, confidence)

        equity = self._equity(bid)
        decision = {
            "timestamp": timestamp,
            "predicted_class": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
            "prob_down": float(probabilities[0]),
            "prob_stationary": float(probabilities[1]),
            "prob_up": float(probabilities[2]),
            "confidence": confidence,
            "threshold": self.policy.model_prob_threshold,
            "action": action,
            "reason": reason,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "execution_price": execution_price,
            "quantity": quantity,
            "fee_eur": fee,
            "cash_eur": self.state.cash_eur,
            "position_qty": self.state.position_qty,
            "equity_eur": equity,
            "pnl_eur": equity - self.policy.initial_cash_eur,
            "open_positions": 1 if self.state.position_qty > 0 else 0,
        }
        self.decisions.append(decision)
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "cash_eur": self.state.cash_eur,
                "position_qty": self.state.position_qty,
                "mark_price": bid,
                "equity_eur": equity,
                "pnl_eur": equity - self.policy.initial_cash_eur,
            }
        )
        return decision

    def summary(self, *, replay: pd.DataFrame, source_files: list[str], start_date: str, duration_minutes: float) -> dict[str, Any]:
        final_bid = float(replay.iloc[-1]["bid_px_1"])
        final_ask = float(replay.iloc[-1]["ask_px_1"])
        final_equity = self._equity(final_bid)
        latest_decision = self.decisions[-1] if self.decisions else {}
        return {
            "status": "ok",
            "provider": "kraken",
            "symbol": self.symbol,
            "model_artifact": str(self.artifact_path),
            "model_device": str(self.device),
            "from_date": start_date,
            "duration_minutes": duration_minutes,
            "source_files": source_files,
            "row_count": int(len(replay)),
            "decision_count": int(len(self.decisions)),
            "trades": len(self.trades),
            "initial_cash_eur": float(self.policy.initial_cash_eur),
            "final_cash_eur": self.state.cash_eur,
            "open_position_qty": self.state.position_qty,
            "mark_to_market_eur": self.state.position_qty * final_bid if final_bid > 0 else 0.0,
            "final_equity_eur": final_equity,
            "pnl_eur": final_equity - self.policy.initial_cash_eur,
            "pnl_pct": (final_equity - self.policy.initial_cash_eur) / self.policy.initial_cash_eur,
            "total_fees_eur": self.state.total_fees_eur,
            "fee_bps": self.policy.fee_bps,
            "slippage_bps": self.policy.slippage_bps,
            "policy": self.policy.to_dict(),
            "latest_market": _latest_book_snapshot(replay.iloc[-1], self.depth_levels),
            "latest_decision": latest_decision,
            "final_bid": final_bid,
            "final_ask": final_ask,
            "note": "Local simulation only. No orders were sent to Kraken.",
        }

    def _try_buy(self, timestamp: str, ask: float, confidence: float) -> tuple[str, str, float, float, float]:
        if self.state.position_qty > 0:
            return "hold", "position_already_open", 0.0, 0.0, 0.0
        if self.policy.max_open_positions < 1:
            return "hold", "max_open_positions_zero", 0.0, 0.0, 0.0
        if self._daily_trade_count(timestamp) >= self.policy.max_trades_per_day:
            return "hold", "max_trades_per_day_reached", 0.0, 0.0, 0.0
        notional = self.state.cash_eur * self.policy.position_fraction
        if notional < self.policy.minimum_order_notional_eur:
            return "hold", "order_below_kraken_minimum", 0.0, 0.0, 0.0
        slippage_fraction = self.policy.slippage_bps / 10000.0
        fee_fraction = self.policy.fee_bps / 10000.0
        execution_price = ask * (1.0 + slippage_fraction)
        fee = notional * fee_fraction
        quantity = max((notional - fee) / execution_price, 0.0)
        if quantity < self.policy.minimum_order_base:
            return "hold", "quantity_below_kraken_minimum", execution_price, quantity, fee
        self.state.cash_eur -= notional
        self.state.position_qty += quantity
        self.state.entry_notional_eur += notional
        self.state.total_fees_eur += fee
        self.trades.append(
            {
                "timestamp": timestamp,
                "side": "buy",
                "price": execution_price,
                "quantity": quantity,
                "notional_eur": notional,
                "fee_eur": fee,
                "confidence": confidence,
            }
        )
        return "buy", "model_up_signal", execution_price, quantity, fee

    def _try_sell(self, timestamp: str, bid: float, confidence: float) -> tuple[str, str, float, float, float]:
        if self.state.position_qty <= 0:
            return "hold", "no_open_position_to_sell", 0.0, 0.0, 0.0
        if self._daily_trade_count(timestamp) >= self.policy.max_trades_per_day:
            return "hold", "max_trades_per_day_reached", 0.0, 0.0, 0.0
        slippage_fraction = self.policy.slippage_bps / 10000.0
        fee_fraction = self.policy.fee_bps / 10000.0
        execution_price = bid * (1.0 - slippage_fraction)
        quantity = self.state.position_qty
        gross = quantity * execution_price
        fee = gross * fee_fraction
        self.state.cash_eur += gross - fee
        self.state.total_fees_eur += fee
        self.state.realized_pnl_eur += gross - fee - self.state.entry_notional_eur
        self.trades.append(
            {
                "timestamp": timestamp,
                "side": "sell",
                "price": execution_price,
                "quantity": quantity,
                "notional_eur": gross,
                "fee_eur": fee,
                "confidence": confidence,
            }
        )
        self.state.position_qty = 0.0
        self.state.entry_notional_eur = 0.0
        return "sell", "model_down_signal", execution_price, quantity, fee

    def _daily_trade_count(self, timestamp: str) -> int:
        day = str(pd.Timestamp(timestamp).date())
        return sum(1 for trade in self.trades if str(pd.Timestamp(trade["timestamp"]).date()) == day)

    def _daily_loss_limit_reached(self, mark_price: float) -> bool:
        if self.policy.max_daily_loss_pct <= 0:
            return False
        loss_fraction = (self._equity(mark_price) - self.policy.initial_cash_eur) / self.policy.initial_cash_eur
        return loss_fraction <= -(self.policy.max_daily_loss_pct / 100.0)

    def _equity(self, mark_price: float) -> float:
        mark_to_market = self.state.position_qty * mark_price if mark_price > 0 else 0.0
        return float(self.state.cash_eur + mark_to_market)


def run_kraken_paper_sim(
    settings: Settings,
    *,
    symbol: str,
    model_artifact: str = "active",
    duration_minutes: float = 60.0,
    from_date: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    replay, source_files, start_date = load_kraken_paper_replay(
        settings,
        symbol=symbol,
        duration_minutes=duration_minutes,
        from_date=from_date,
    )
    artifact_path, artifact_payload = _resolve_artifact(settings, model_artifact)
    latest_price = _latest_mid_price(replay)
    policy = build_kraken_paper_policy(settings, symbol=symbol, latest_price=latest_price)
    simulator = KrakenPaperSimulator(
        settings=settings,
        symbol=symbol,
        artifact_path=artifact_path,
        artifact_payload=artifact_payload,
        policy=policy,
    )
    simulator.run(replay)
    payload = simulator.summary(
        replay=replay,
        source_files=source_files,
        start_date=start_date,
        duration_minutes=duration_minutes,
    )
    payload.update(_write_kraken_paper_report(settings, symbol=symbol, simulator=simulator, payload=payload, run_id=run_id))
    return payload


def load_kraken_paper_replay(
    settings: Settings,
    *,
    symbol: str,
    duration_minutes: float,
    from_date: str | None = None,
) -> tuple[pd.DataFrame, list[str], str]:
    start_date = from_date or _latest_available_date(settings, symbol)
    frame, source_files = load_lob_capture_frame(
        settings,
        symbol=symbol,
        provider="kraken",
        from_date=start_date,
    )
    if frame.empty:
        raise ValueError(f"No Kraken LOB data available for {symbol} from {start_date}.")
    frame = frame.sort_values(["event_ts_utc", "event_index"]).reset_index(drop=True)
    cutoff = frame["event_ts_utc"].max() - pd.Timedelta(minutes=float(duration_minutes))
    replay = frame[frame["event_ts_utc"] >= cutoff].reset_index(drop=True)
    return replay, source_files, start_date


def build_kraken_paper_policy(
    settings: Settings,
    *,
    symbol: str,
    latest_price: float | None = None,
    market_minimum_fetcher: Callable[[str, float | None], KrakenMarketMinimum] | None = None,
) -> KrakenPaperPolicy:
    mode = str(settings.kraken_lob.paper_initial_cash_mode).strip().lower()
    position_fraction = min(max(float(settings.kraken_lob.paper_position_fraction), 0.01), 1.0)
    warnings: list[str] = []
    if mode == "fixed":
        minimum = _fallback_market_minimum(symbol, latest_price)
        initial_cash = float(settings.kraken_lob.paper_initial_cash_eur)
        initial_cash_source = "configured_fixed"
    else:
        fetcher = market_minimum_fetcher or fetch_kraken_market_minimum
        try:
            minimum = fetcher(symbol, latest_price)
        except Exception as exc:
            minimum = _fallback_market_minimum(symbol, latest_price)
            warnings.append(f"Kraken minimum lookup failed; using fallback minimum. Root cause: {exc}")
        if minimum.warning:
            warnings.append(minimum.warning)
        buffer_fraction = float(settings.kraken_lob.paper_min_cash_buffer_bps) / 10000.0
        initial_cash = (minimum.minimum_order_notional_eur / position_fraction) * (1.0 + buffer_fraction)
        initial_cash_source = "dynamic_kraken_minimum"
    return KrakenPaperPolicy(
        initial_cash_eur=float(initial_cash),
        initial_cash_mode=mode,
        initial_cash_source=initial_cash_source,
        minimum_order_base=float(minimum.minimum_order_base),
        minimum_order_notional_eur=float(minimum.minimum_order_notional_eur),
        minimum_order_source=minimum.source,
        position_fraction=position_fraction,
        fee_bps=float(settings.kraken_lob.paper_fee_bps),
        slippage_bps=float(settings.kraken_lob.paper_slippage_bps),
        model_prob_threshold=float(settings.models.model_prob_threshold),
        max_trades_per_day=int(settings.risk.max_trades_per_day),
        max_daily_loss_pct=float(settings.risk.max_daily_loss_pct),
        max_open_positions=int(settings.risk.max_open_positions),
        warnings=tuple(warnings),
    )


def fetch_kraken_market_minimum(symbol: str, latest_price: float | None = None) -> KrakenMarketMinimum:
    candidates = _kraken_pair_candidates(symbol)
    last_error: Exception | None = None
    for pair in candidates:
        try:
            asset_payload = _kraken_public_get("AssetPairs", {"pair": pair})
            result = asset_payload.get("result") or {}
            if not result:
                continue
            pair_key, pair_info = next(iter(result.items()))
            minimum_order = float(pair_info.get("ordermin") or 0.0)
            ticker_payload = _kraken_public_get("Ticker", {"pair": pair_key})
            ticker_result = ticker_payload.get("result") or {}
            ticker_info = ticker_result.get(pair_key) or next(iter(ticker_result.values()))
            price = float((ticker_info.get("c") or [latest_price or 0.0])[0])
            if minimum_order > 0 and price > 0:
                return KrakenMarketMinimum(
                    symbol=symbol.upper(),
                    pair_key=pair_key,
                    minimum_order_base=minimum_order,
                    price_eur=price,
                    source="kraken_public_asset_pairs",
                )
        except Exception as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise ValueError(f"Kraken did not return tradable pair metadata for {symbol}.")


def _kraken_public_get(endpoint: str, params: dict[str, str], *, timeout: float = 10.0) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"https://api.kraken.com/0/public/{endpoint}?{query}"
    context = ssl.create_default_context()
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    request = urllib.request.Request(url, headers={"User-Agent": "MicroAlpha-IBKR/kraken-paper-ui"})
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    errors = payload.get("error") or []
    if errors:
        raise ValueError(f"Kraken API returned errors for {endpoint}: {errors}")
    return payload


def _fallback_market_minimum(symbol: str, latest_price: float | None = None) -> KrakenMarketMinimum:
    base, _quote = _split_symbol(symbol)
    price = float(latest_price or 0.0)
    if price <= 0:
        price = 1.0
    minimum = BTC_MIN_ORDER_FALLBACK if base in {"BTC", "XBT"} else BTC_MIN_ORDER_FALLBACK
    return KrakenMarketMinimum(
        symbol=symbol.upper(),
        pair_key=_kraken_pair_candidates(symbol)[0],
        minimum_order_base=minimum,
        price_eur=price,
        source="fallback_btc_minimum",
        warning="Using fallback BTC minimum order size because Kraken public metadata was unavailable.",
    )


def _write_kraken_paper_report(
    settings: Settings,
    *,
    symbol: str,
    simulator: KrakenPaperSimulator,
    payload: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, str]:
    token = run_id or datetime.now(timezone.utc).strftime("kraken-paper-%Y%m%dT%H%M%S")
    report_root = Path(settings.lob_capture.report_root) / "kraken_paper" / _symbol_path_token(symbol) / token
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "summary.json"
    trades_path = report_root / "trades.csv"
    decisions_path = report_root / "decisions.csv"
    equity_path = report_root / "equity.csv"
    state_path = report_root / "state.json"
    payload_with_paths = {
        **payload,
        "run_id": token,
        "report_dir": str(report_root),
        "report_path": str(report_path),
        "trades_path": str(trades_path),
        "decisions_path": str(decisions_path),
        "equity_path": str(equity_path),
        "state_path": str(state_path),
    }
    report_path.write_text(json.dumps(payload_with_paths, indent=2, sort_keys=True, default=str), encoding="utf-8")
    pd.DataFrame(simulator.trades).to_csv(trades_path, index=False)
    pd.DataFrame(simulator.decisions).to_csv(decisions_path, index=False)
    pd.DataFrame(simulator.equity_curve).to_csv(equity_path, index=False)
    state_path.write_text(json.dumps(asdict(simulator.state), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "run_id": token,
        "report_dir": str(report_root),
        "report_path": str(report_path),
        "trades_path": str(trades_path),
        "decisions_path": str(decisions_path),
        "equity_path": str(equity_path),
        "state_path": str(state_path),
    }


def _resolve_artifact(settings: Settings, model_artifact: str) -> tuple[Path, dict[str, Any]]:
    registry = ModelRegistry(settings.models.registry_path)
    if model_artifact == "active":
        record = registry.get_active_model("deep")
        if not record:
            raise ValueError("No active deep model is registered.")
        path = Path(str(record["artifact_path"]))
    else:
        candidate = Path(model_artifact)
        if candidate.exists():
            path = candidate
        else:
            record = registry.get_model("deep", model_artifact)
            if not record:
                raise ValueError(f"Deep model artifact not found: {model_artifact}")
            path = Path(str(record["artifact_path"]))
    payload = torch.load(path, map_location="cpu")
    if payload.get("model_family") != "deep_lob_reference_like":
        raise ValueError("Kraken paper simulation requires a DeepLOB-like artifact.")
    return path, payload


def _latest_available_date(settings: Settings, symbol: str) -> str:
    root = Path(settings.kraken_lob.output_root) / _symbol_path_token(symbol)
    dates = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.exists() else []
    if not dates:
        raise ValueError(f"No Kraken LOB capture directories found for {symbol}.")
    return dates[-1]


def _normalize_window(window: pd.DataFrame, feature_columns: list[str], depth_levels: int) -> np.ndarray:
    raw_values = window[feature_columns].to_numpy(dtype=np.float32).copy()
    last_mid = float((window.iloc[-1]["bid_px_1"] + window.iloc[-1]["ask_px_1"]) / 2.0)
    last_mid = max(last_mid, 1e-9)
    price_count = depth_levels * 2
    price_window = raw_values[:, :price_count]
    size_window = raw_values[:, price_count:]
    raw_values[:, :price_count] = ((price_window / last_mid) - 1.0) * 10000.0
    raw_values[:, price_count:] = np.log1p(np.clip(size_window, a_min=0.0, a_max=None))
    return raw_values


def _latest_mid_price(frame: pd.DataFrame) -> float:
    bid = float(frame.iloc[-1]["bid_px_1"])
    ask = float(frame.iloc[-1]["ask_px_1"])
    return (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask, 0.0)


def _latest_book_snapshot(row: pd.Series, depth_levels: int) -> dict[str, Any]:
    levels = []
    for level in range(1, depth_levels + 1):
        bid_px = float(row.get(f"bid_px_{level}", 0.0) or 0.0)
        ask_px = float(row.get(f"ask_px_{level}", 0.0) or 0.0)
        bid_sz = float(row.get(f"bid_sz_{level}", 0.0) or 0.0)
        ask_sz = float(row.get(f"ask_sz_{level}", 0.0) or 0.0)
        levels.append(
            {
                "level": level,
                "bid_px": bid_px,
                "bid_sz": bid_sz,
                "ask_px": ask_px,
                "ask_sz": ask_sz,
            }
        )
    bid = levels[0]["bid_px"] if levels else 0.0
    ask = levels[0]["ask_px"] if levels else 0.0
    return {
        "timestamp": str(row.get("event_ts_utc", "")),
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0,
        "spread_bps": ((ask - bid) / ((ask + bid) / 2.0) * 10000.0) if bid > 0 and ask > 0 else 0.0,
        "levels": levels,
    }


def _kraken_pair_candidates(symbol: str) -> list[str]:
    base, quote = _split_symbol(symbol)
    base_alias = "XBT" if base == "BTC" else base
    candidates = [
        f"{base}{quote}",
        f"{base_alias}{quote}",
        f"X{base_alias}Z{quote}",
        symbol.upper(),
    ]
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _split_symbol(symbol: str) -> tuple[str, str]:
    normalized = str(symbol).upper().replace("-", "/")
    if "/" not in normalized:
        raise ValueError(f"Expected Kraken symbol like BTC/EUR, got {symbol!r}.")
    base, quote = normalized.split("/", maxsplit=1)
    return base, quote


def _symbol_path_token(symbol: str) -> str:
    return str(symbol).upper().replace("/", "_").replace("-", "_").replace(" ", "_")
