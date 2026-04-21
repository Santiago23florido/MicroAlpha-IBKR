from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config import Settings
from data.lob_dataset import load_lob_capture_frame
from models.deep_lob import DeepLOBReferenceLike, lob_feature_columns
from models.registry import ModelRegistry


def run_kraken_paper_sim(
    settings: Settings,
    *,
    symbol: str,
    model_artifact: str = "active",
    duration_minutes: float = 60.0,
    from_date: str | None = None,
) -> dict[str, Any]:
    artifact_path, artifact_payload = _resolve_artifact(settings, model_artifact)
    depth_levels = int(artifact_payload.get("depth_levels", settings.models.lob_depth_levels))
    sequence_length = int(artifact_payload.get("sequence_length", settings.models.lob_sequence_length))
    model = DeepLOBReferenceLike(
        feature_dim=depth_levels * 4,
        sequence_length=sequence_length,
    )
    model.load_state_dict(artifact_payload["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

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
    if len(replay) < sequence_length:
        raise ValueError(
            f"Not enough rows for paper simulation. Need at least {sequence_length}, got {len(replay)}."
        )

    cash = float(settings.kraken_lob.paper_initial_cash_eur)
    position_qty = 0.0
    entry_value = 0.0
    trades: list[dict[str, Any]] = []
    fee_fraction = float(settings.kraken_lob.paper_fee_bps) / 10000.0
    slippage_fraction = float(settings.kraken_lob.paper_slippage_bps) / 10000.0
    threshold = float(settings.models.model_prob_threshold)
    feature_columns = lob_feature_columns(depth_levels)

    for end_index in range(sequence_length - 1, len(replay)):
        window = replay.iloc[end_index - sequence_length + 1 : end_index + 1]
        features = _normalize_window(window, feature_columns, depth_levels)
        with torch.no_grad():
            logits, _predicted_return = model(torch.tensor(features[None, :, :], dtype=torch.float32, device=device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        row = replay.iloc[end_index]
        timestamp = str(row["event_ts_utc"])
        bid = float(row["bid_px_1"])
        ask = float(row["ask_px_1"])
        if bid <= 0 or ask <= 0 or confidence < threshold:
            continue

        if predicted_class == 2 and position_qty <= 0 and cash > 10.0:
            notional = cash * 0.25
            execution_price = ask * (1.0 + slippage_fraction)
            fee = notional * fee_fraction
            quantity = max((notional - fee) / execution_price, 0.0)
            if quantity <= 0:
                continue
            cash -= notional
            position_qty += quantity
            entry_value += notional
            trades.append(
                {
                    "timestamp": timestamp,
                    "side": "buy",
                    "price": execution_price,
                    "quantity": quantity,
                    "fee": fee,
                    "confidence": confidence,
                }
            )
        elif predicted_class == 0 and position_qty > 0:
            execution_price = bid * (1.0 - slippage_fraction)
            gross = position_qty * execution_price
            fee = gross * fee_fraction
            cash += gross - fee
            trades.append(
                {
                    "timestamp": timestamp,
                    "side": "sell",
                    "price": execution_price,
                    "quantity": position_qty,
                    "fee": fee,
                    "confidence": confidence,
                }
            )
            position_qty = 0.0
            entry_value = 0.0

    final_bid = float(replay.iloc[-1]["bid_px_1"])
    mark_to_market = position_qty * final_bid if final_bid > 0 else 0.0
    final_equity = cash + mark_to_market
    pnl = final_equity - float(settings.kraken_lob.paper_initial_cash_eur)
    report_root = Path(settings.lob_capture.report_root) / "kraken_paper" / _symbol_path_token(symbol)
    report_root.mkdir(parents=True, exist_ok=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = report_root / f"paper_sim_{token}.json"
    trades_path = report_root / f"paper_sim_{token}.csv"
    payload = {
        "status": "ok",
        "provider": "kraken",
        "symbol": symbol.upper(),
        "model_artifact": str(artifact_path),
        "from_date": start_date,
        "duration_minutes": duration_minutes,
        "source_files": source_files,
        "row_count": int(len(replay)),
        "trades": len(trades),
        "initial_cash_eur": float(settings.kraken_lob.paper_initial_cash_eur),
        "final_cash_eur": cash,
        "open_position_qty": position_qty,
        "mark_to_market_eur": mark_to_market,
        "final_equity_eur": final_equity,
        "pnl_eur": pnl,
        "pnl_pct": pnl / float(settings.kraken_lob.paper_initial_cash_eur),
        "fee_bps": float(settings.kraken_lob.paper_fee_bps),
        "slippage_bps": float(settings.kraken_lob.paper_slippage_bps),
        "report_path": str(report_path),
        "trades_path": str(trades_path),
        "note": "Local simulation only. No orders were sent to Kraken.",
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(trades).to_csv(trades_path, index=False)
    return payload


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


def _symbol_path_token(symbol: str) -> str:
    return str(symbol).upper().replace("/", "_").replace("-", "_").replace(" ", "_")
