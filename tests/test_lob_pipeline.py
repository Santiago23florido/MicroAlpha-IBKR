from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import load_settings
from data.lob_dataset import build_lob_dataset
from ingestion.lob_capture import run_lob_capture_loop
from ingestion.lob_reconstruction import LOBBookState, LOBDepthUpdate
from models.train_deep import evaluate_deep_daily, train_deep_model


class FakeDepthClient:
    def __init__(self, batches: list[list[dict[str, object]]]) -> None:
        self._batches = list(batches)
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def subscribe_market_depth(
        self,
        *,
        symbol: str,
        num_rows: int = 10,
        exchange: str | None = None,
        currency: str | None = None,
        is_smart_depth: bool = True,
    ) -> int:
        return 1

    def consume_market_depth_events(
        self,
        req_id: int,
        *,
        timeout: float = 1.0,
        max_events: int = 500,
    ) -> list[dict[str, object]]:
        if self._batches:
            return self._batches.pop(0)
        return []

    def cancel_market_depth(self, req_id: int) -> None:
        return None


def test_lob_book_state_handles_insert_update_delete_and_reset() -> None:
    book = LOBBookState(symbol="SPY", depth_levels=2)
    insert_bid = LOBDepthUpdate("SPY", "2026-04-21T13:30:00Z", 0, 0, 1, 100.0, 10.0)
    insert_ask = LOBDepthUpdate("SPY", "2026-04-21T13:30:01Z", 0, 0, 0, 100.1, 12.0)
    update_bid = LOBDepthUpdate("SPY", "2026-04-21T13:30:02Z", 0, 1, 1, 100.02, 15.0)
    delete_ask = LOBDepthUpdate("SPY", "2026-04-21T13:30:03Z", 0, 2, 0, 0.0, 0.0)

    snapshot = book.apply(insert_bid)
    assert snapshot["bid_px_1"] == 100.0
    snapshot = book.apply(insert_ask)
    assert snapshot["ask_px_1"] == 100.1
    snapshot = book.apply(update_bid)
    assert snapshot["bid_px_1"] == 100.02
    snapshot = book.apply(delete_ask)
    assert snapshot["ask_px_1"] == 0.0

    book.reset()
    assert book.reset_count == 1
    assert book.bids == []
    assert book.asks == []


def test_run_lob_capture_loop_persists_chunks_and_state(tmp_path: Path) -> None:
    settings = _build_temp_settings(tmp_path)
    client = FakeDepthClient(
        batches=[
            [
                {"event_type": "depth_update", "timestamp_utc": "2026-04-21T13:30:00Z", "position": 0, "operation": 0, "side": 1, "price": 100.0, "size": 10.0, "market_maker": "A", "is_smart_depth": True, "source": "updateMktDepthL2"},
                {"event_type": "depth_update", "timestamp_utc": "2026-04-21T13:30:00Z", "position": 0, "operation": 0, "side": 0, "price": 100.1, "size": 12.0, "market_maker": "B", "is_smart_depth": True, "source": "updateMktDepthL2"},
            ],
            [
                {"event_type": "depth_update", "timestamp_utc": "2026-04-21T13:30:01Z", "position": 1, "operation": 0, "side": 1, "price": 99.99, "size": 20.0, "market_maker": "A", "is_smart_depth": True, "source": "updateMktDepthL2"},
                {"event_type": "depth_update", "timestamp_utc": "2026-04-21T13:30:01Z", "position": 1, "operation": 0, "side": 0, "price": 100.11, "size": 18.0, "market_maker": "B", "is_smart_depth": True, "source": "updateMktDepthL2"},
            ],
        ]
    )

    payload = run_lob_capture_loop(
        settings,
        symbol="SPY",
        levels=2,
        session_id="lob-test-session",
        rth_only=False,
        client=client,
        max_events=4,
    )

    assert payload["status"] == "stopped"
    chunk_files = list((Path(settings.lob_capture.output_root) / "SPY").glob("**/chunks/*.parquet"))
    assert chunk_files
    state_path = Path(settings.lob_capture.state_root) / "SPY.json"
    assert state_path.exists()
    state = pd.read_json(state_path, typ="series")
    assert int(state["row_count"]) == 4


def test_build_lob_dataset_and_train_deep_lob(tmp_path: Path) -> None:
    settings = _build_temp_settings(tmp_path)
    raw_root = Path(settings.lob_capture.output_root) / "SPY" / "2026-04-21" / "chunks"
    raw_root.mkdir(parents=True, exist_ok=True)
    frame = _build_lob_raw_frame("2026-04-21", rows=12)
    frame.to_parquet(raw_root / "lob_chunk.parquet", index=False)

    dataset_result = build_lob_dataset(
        settings,
        symbol="SPY",
        from_date="2026-04-21",
        horizon_events=1,
    )
    assert dataset_result["status"] == "ok"
    dataset_path = Path(dataset_result["dataset_path"])
    assert dataset_path.exists()

    payload = train_deep_model(
        settings,
        data_path=str(dataset_path),
        model_name="deep_lob_reference_like",
        epochs=1,
        set_active=False,
    )
    assert payload["metadata"]["dataset_type"] == "ibkr_lob_depth"
    assert payload["record"]["model_family"] == "deep_lob_reference_like"
    assert Path(payload["record"]["artifact_path"]).exists()


def test_evaluate_deep_daily_walk_forward(tmp_path: Path) -> None:
    settings = _build_temp_settings(tmp_path)
    symbol_root = Path(settings.lob_capture.output_root) / "SPY"
    for session_date in ["2026-04-21", "2026-04-22", "2026-04-23"]:
        chunk_dir = symbol_root / session_date / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        _build_lob_raw_frame(session_date, rows=10).to_parquet(chunk_dir / "lob_chunk.parquet", index=False)

    payload = evaluate_deep_daily(
        settings,
        symbol="SPY",
        from_date="2026-04-21",
        epochs=1,
    )
    assert payload["status"] == "ok"
    assert payload["daily_metrics"]
    assert Path(payload["report_path"]).exists()


def _build_temp_settings(tmp_path: Path):
    base_settings = load_settings(".env", config_dir="config")
    data_root = (tmp_path / "data").resolve()
    log_dir = data_root / "logs"
    model_artifacts_dir = data_root / "models" / "artifacts"
    report_dir = data_root / "reports"
    for path in [log_dir, model_artifacts_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)
    paths = replace(
        base_settings.paths,
        data_root=str(data_root),
        processed_dir=str((data_root / "processed").resolve()),
        model_dir=str((data_root / "models").resolve()),
        model_artifacts_dir=str(model_artifacts_dir.resolve()),
        log_dir=str(log_dir.resolve()),
        report_dir=str(report_dir.resolve()),
    )
    models = replace(
        base_settings.models,
        artifacts_dir=str(model_artifacts_dir.resolve()),
        registry_path=str((model_artifacts_dir / "registry.json").resolve()),
        lob_sequence_length=3,
        lob_depth_levels=2,
        lob_horizon_events=1,
        lob_train_batch_size=4,
        lob_eval_batch_size=4,
    )
    storage = replace(
        base_settings.storage,
        log_file=str((log_dir / "microalpha.log").resolve()),
        execution_log_file=str((report_dir / "executions.csv").resolve()),
        runtime_db_path=str((data_root / "processed" / "runtime" / "microalpha.db").resolve()),
    )
    lob_capture = replace(
        base_settings.lob_capture,
        output_root=str((data_root / "raw" / "ibkr_lob").resolve()),
        state_root=str((data_root / "processed" / "ibkr_lob_state").resolve()),
        session_root=str((data_root / "processed" / "ibkr_lob_sessions").resolve()),
        dataset_root=str((data_root / "processed" / "lob_datasets").resolve()),
        report_root=str((report_dir / "lob").resolve()),
        depth_levels=2,
        batch_size=2,
        flush_interval_seconds=0.1,
    )
    return replace(base_settings, paths=paths, models=models, storage=storage, lob_capture=lob_capture)


def _build_lob_raw_frame(session_date: str, *, rows: int) -> pd.DataFrame:
    base_timestamp = pd.Timestamp(f"{session_date}T13:30:00Z")
    frame = pd.DataFrame(
        {
            "symbol": ["SPY"] * rows,
            "capture_session_id": [f"lob-{session_date}"] * rows,
            "event_ts_utc": [base_timestamp + pd.Timedelta(seconds=index) for index in range(rows)],
            "exchange_ts": [base_timestamp + pd.Timedelta(seconds=index) for index in range(rows)],
            "session_date": [session_date] * rows,
            "session_window": ["primary"] * rows,
            "provider": ["ibkr"] * rows,
            "source": ["ibkr_market_depth"] * rows,
            "rth_only": [True] * rows,
            "observed_bid_levels": [2] * rows,
            "observed_ask_levels": [2] * rows,
            "event_index": list(range(1, rows + 1)),
            "reset_count": [0] * rows,
            "bid_px_1": [100.00 + index * 0.01 for index in range(rows)],
            "ask_px_1": [100.02 + index * 0.01 for index in range(rows)],
            "bid_px_2": [99.99 + index * 0.01 for index in range(rows)],
            "ask_px_2": [100.03 + index * 0.01 for index in range(rows)],
            "bid_sz_1": [10.0 + index for index in range(rows)],
            "ask_sz_1": [11.0 + index for index in range(rows)],
            "bid_sz_2": [20.0 + index for index in range(rows)],
            "ask_sz_2": [21.0 + index for index in range(rows)],
        }
    )
    return frame
