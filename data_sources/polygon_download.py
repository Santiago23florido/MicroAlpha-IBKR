from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from config.polygon import load_polygon_config
from data_sources.polygon_client import PolygonClient
from data_sources.polygon_normalizer import normalize_polygon_frame
from data_sources.training_dataset_export import export_training_dataset


def fetch_training_data(
    settings: Settings,
    *,
    provider: str,
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    output_path: str | Path,
) -> dict[str, Any]:
    _validate_provider(provider)
    config = load_polygon_config(settings)
    client = PolygonClient(config)
    raw = client.fetch_aggregates(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval)
    normalized, metadata = normalize_polygon_frame(
        raw,
        symbol=symbol,
        interval=interval,
        config=config,
        source_mode="api",
    )
    export_result = export_training_dataset(
        normalized,
        output_path=output_path,
        metadata=metadata,
        write_parquet=config.write_parquet,
        write_manifest=config.write_manifest,
    )
    return {
        "status": "ok",
        "provider": "polygon",
        "mode": "api",
        "symbol": symbol.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "metadata": metadata,
        **export_result,
    }


def normalize_training_data(
    settings: Settings,
    *,
    provider: str,
    input_path: str | Path,
    output_path: str | Path,
    symbol: str | None = None,
    interval: str | None = None,
) -> dict[str, Any]:
    _validate_provider(provider)
    config = load_polygon_config(settings)
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Manual Polygon CSV file not found: {source}")
    frame = _read_frame(source)
    normalized, metadata = normalize_polygon_frame(
        frame,
        symbol=symbol or config.default_symbol,
        interval=interval or config.default_interval,
        config=config,
        source_mode="manual_csv",
    )
    export_result = export_training_dataset(
        normalized,
        output_path=output_path,
        metadata={**metadata, "input_path": str(source)},
        write_parquet=config.write_parquet,
        write_manifest=config.write_manifest,
    )
    return {
        "status": "ok",
        "provider": "polygon",
        "mode": "manual_csv",
        "input_path": str(source),
        "metadata": metadata,
        **export_result,
    }


def prepare_training_data(
    settings: Settings,
    *,
    provider: str,
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str | None = None,
    output_path: str | Path | None = None,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    _validate_provider(provider)
    config = load_polygon_config(settings)
    resolved_output_path = Path(output_path) if output_path else Path(config.output_root) / _default_filename(
        symbol=symbol or config.default_symbol,
        interval=interval or config.default_interval,
        mode="manual" if input_path else "api",
    )
    if input_path:
        return normalize_training_data(
            settings,
            provider=provider,
            input_path=input_path,
            output_path=resolved_output_path,
            symbol=symbol or config.default_symbol,
            interval=interval or config.default_interval,
        )
    return fetch_training_data(
        settings,
        provider=provider,
        symbol=symbol or config.default_symbol,
        start_date=start_date or config.default_start_date,
        end_date=end_date or config.default_end_date,
        interval=interval or config.default_interval,
        output_path=resolved_output_path,
    )


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _validate_provider(provider: str) -> None:
    if str(provider).strip().lower() != "polygon":
        raise ValueError("Only provider=polygon is supported by the bootstrap training-data integration.")


def _default_filename(*, symbol: str, interval: str, mode: str) -> str:
    suffix = "manual_training" if mode == "manual" else "training"
    return f"{str(symbol).upper()}_{interval}_{suffix}.csv"
