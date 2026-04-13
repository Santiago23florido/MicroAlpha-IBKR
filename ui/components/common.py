from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_kv_table(title: str, payload: dict[str, Any]) -> None:
    st.subheader(title)
    frame = pd.DataFrame(
        [{"key": key, "value": value} for key, value in payload.items()]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_records_table(title: str, records: list[dict[str, Any]]) -> None:
    st.subheader(title)
    if not records:
        st.info("No records available.")
        return
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
