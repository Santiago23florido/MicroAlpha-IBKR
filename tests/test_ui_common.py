from __future__ import annotations

import pandas as pd

from ui.components.common import _prepare_frame_for_streamlit


def test_prepare_frame_for_streamlit_normalizes_mixed_object_columns() -> None:
    frame = pd.DataFrame(
        [
            {"key": "connected", "value": True, "payload": {"a": 1}},
            {"key": "message", "value": "ok", "payload": None},
        ]
    )

    prepared = _prepare_frame_for_streamlit(frame)

    assert prepared["value"].tolist() == ["True", "ok"]
    assert prepared["payload"].tolist() == ['{"a": 1}', ""]
