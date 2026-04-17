from __future__ import annotations

import pandas as pd

from config import Settings
from features.definitions import IndicatorDefinition, IndicatorDependency
from features.indicators._utils import rolling_mean_by_group, safe_divide, session_groupby


def build_microstructure_indicator_definitions() -> list[IndicatorDefinition]:
    dependencies = (
        IndicatorDependency("bid", ("bid",)),
        IndicatorDependency("ask", ("ask",)),
    )
    sized_dependencies = dependencies + (
        IndicatorDependency("bid_size", ("bid_size",)),
        IndicatorDependency("ask_size", ("ask_size",)),
    )
    return [
        IndicatorDefinition(
            name="spread",
            family="microstructure",
            description="Absolute bid-ask spread.",
            required_inputs=dependencies,
            output_columns=("spread",),
            output_type="numeric",
            calculator=_calc_microstructure_bundle,
        ),
        IndicatorDefinition(
            name="spread_bps",
            family="microstructure",
            description="Bid-ask spread in basis points.",
            required_inputs=dependencies,
            output_columns=("spread_bps", "rolling_spread_mean_bps", "rolling_spread_std_bps"),
            output_type="numeric",
            calculator=_calc_microstructure_bundle,
        ),
        IndicatorDefinition(
            name="mid_price",
            family="microstructure",
            description="Mid quote.",
            required_inputs=dependencies,
            output_columns=("mid_price",),
            output_type="numeric",
            calculator=_calc_microstructure_bundle,
        ),
        IndicatorDefinition(
            name="weighted_mid_price",
            family="microstructure",
            description="Depth-weighted mid quote.",
            required_inputs=sized_dependencies,
            output_columns=("weighted_mid_price",),
            output_type="numeric",
            calculator=_calc_microstructure_bundle,
        ),
        IndicatorDefinition(
            name="imbalance",
            family="microstructure",
            description="Bid/ask size imbalance.",
            required_inputs=sized_dependencies,
            output_columns=("imbalance",),
            output_type="numeric",
            calculator=_calc_microstructure_bundle,
        ),
        IndicatorDefinition(
            name="rolling_imbalance",
            family="microstructure",
            description="Rolling imbalance mean and standard deviation.",
            required_inputs=sized_dependencies,
            output_columns=("rolling_imbalance_mean", "rolling_imbalance_std"),
            output_type="numeric",
            calculator=_calc_microstructure_bundle,
        ),
        IndicatorDefinition(
            name="delta_imbalance",
            family="microstructure",
            description="One-step change in order-book imbalance.",
            required_inputs=sized_dependencies,
            output_columns=("delta_imbalance",),
            output_type="numeric",
            calculator=_calc_microstructure_bundle,
        ),
        IndicatorDefinition(
            name="total_depth",
            family="microstructure",
            description="Visible depth from bid and ask sizes.",
            required_inputs=sized_dependencies,
            output_columns=("total_depth",),
            output_type="numeric",
            calculator=_calc_microstructure_bundle,
        ),
        IndicatorDefinition(
            name="depth_ratio",
            family="microstructure",
            description="Bid to ask depth ratio.",
            required_inputs=sized_dependencies,
            output_columns=("depth_ratio",),
            output_type="numeric",
            calculator=_calc_microstructure_bundle,
        ),
        IndicatorDefinition(
            name="microprice_proxy",
            family="microstructure",
            description="Microprice from quotes and depth.",
            required_inputs=sized_dependencies,
            output_columns=("microprice_proxy",),
            output_type="numeric",
            calculator=_calc_microstructure_bundle,
        ),
    ]


def _calc_microstructure_bundle(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    bid = frame["bid"]
    ask = frame["ask"]
    bid_size = frame["bid_size"] if "bid_size" in frame.columns else pd.Series(0.0, index=frame.index)
    ask_size = frame["ask_size"] if "ask_size" in frame.columns else pd.Series(0.0, index=frame.index)

    mid = (bid + ask) / 2.0
    spread = ask - bid
    spread_bps = safe_divide(spread, mid) * 10000.0
    total_depth = bid_size.fillna(0.0) + ask_size.fillna(0.0)
    weighted_mid = safe_divide((bid * bid_size) + (ask * ask_size), total_depth)
    imbalance = safe_divide(bid_size - ask_size, total_depth).fillna(0.0)
    microprice = safe_divide((ask * bid_size) + (bid * ask_size), total_depth)
    depth_ratio = safe_divide(bid_size, ask_size)

    temp = frame.copy()
    temp["_imbalance"] = imbalance
    temp["_spread_bps"] = spread_bps
    rolling_window = int(params.get("window", settings.feature_pipeline.rolling_short_window))
    rolling_imbalance_mean = session_groupby(temp)["_imbalance"].transform(
        lambda series: series.rolling(rolling_window, min_periods=1).mean()
    )
    rolling_imbalance_std = session_groupby(temp)["_imbalance"].transform(
        lambda series: series.rolling(rolling_window, min_periods=1).std()
    )
    delta_imbalance = session_groupby(temp)["_imbalance"].transform(lambda series: series.diff())
    rolling_spread_mean = session_groupby(temp)["_spread_bps"].transform(
        lambda series: series.rolling(rolling_window, min_periods=1).mean()
    )
    rolling_spread_std = session_groupby(temp)["_spread_bps"].transform(
        lambda series: series.rolling(rolling_window, min_periods=1).std()
    )

    return pd.DataFrame(
        {
            "spread": spread,
            "spread_bps": spread_bps,
            "rolling_spread_mean_bps": rolling_spread_mean,
            "rolling_spread_std_bps": rolling_spread_std,
            "mid_price": mid,
            "weighted_mid_price": weighted_mid.fillna(mid),
            "imbalance": imbalance,
            "rolling_imbalance_mean": rolling_imbalance_mean,
            "rolling_imbalance_std": rolling_imbalance_std,
            "delta_imbalance": delta_imbalance,
            "total_depth": total_depth,
            "depth_ratio": depth_ratio,
            "microprice_proxy": microprice.fillna(mid),
        },
        index=frame.index,
    )
