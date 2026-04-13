from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def render_kv_table(title: str, payload: dict[str, Any]) -> None:
    st.subheader(title)
    frame = _prepare_frame_for_streamlit(pd.DataFrame(
        [{"key": key, "value": value} for key, value in payload.items()]
    ))
    st.dataframe(frame, width="stretch", hide_index=True)


def render_records_table(title: str, records: list[dict[str, Any]]) -> None:
    st.subheader(title)
    if not records:
        st.info("No records available.")
        return
    frame = _prepare_frame_for_streamlit(pd.DataFrame(records))
    st.dataframe(frame, width="stretch", hide_index=True)


def _prepare_frame_for_streamlit(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(_normalize_cell_value)
    return normalized


def _normalize_cell_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    if isinstance(value, (datetime, date, time, Path)):
        return str(value)
    return str(value)
