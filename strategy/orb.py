from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from config import Settings
from data.schemas import MarketSnapshot, ORBState
from engine.market_clock import MarketClock, MarketClockState


@dataclass(frozen=True)
class OpeningRange:
    high: float | None
    low: float | None
    mid: float | None
    width: float | None
    complete: bool


class OpeningRangeBreakoutStrategy:
    def __init__(self, settings: Settings, market_clock: MarketClock) -> None:
        self.settings = settings
        self.market_clock = market_clock

    def evaluate(
        self,
        *,
        symbol: str,
        bars: pd.DataFrame,
        market_snapshot: MarketSnapshot,
        reference_time: datetime,
    ) -> ORBState:
        clock_state = self.market_clock.get_market_state(reference_time)
        opening_range = self._compute_opening_range(bars, clock_state)
        current_price = (
            market_snapshot.last
            or market_snapshot.ask
            or market_snapshot.bid
            or opening_range.mid
            or 0.0
        )

        breakout_direction: str | None = None
        breakout_distance: float | None = None
        candidate_reason = "No ORB breakout candidate is active."
        no_trade_reason: str | None = None

        if not opening_range.complete:
            no_trade_reason = "Opening range is not complete yet."
        elif current_price > (opening_range.high or float("inf")):
            breakout_direction = "long"
            breakout_distance = current_price - float(opening_range.high or current_price)
            candidate_reason = "ORB breakout detected above opening range high."
        elif current_price < (opening_range.low or float("-inf")):
            breakout_direction = "short"
            breakout_distance = float(opening_range.low or current_price) - current_price
            candidate_reason = "ORB breakout detected below opening range low."
            if not self.settings.trading.allow_shorts:
                no_trade_reason = "Short breakout detected but ALLOW_SHORTS is false."
        else:
            no_trade_reason = "Price remains inside the opening range."

        trading_allowed = opening_range.complete and (
            clock_state.is_primary_window or clock_state.is_secondary_window
        )
        if not trading_allowed and no_trade_reason is None:
            if clock_state.session_window == "opening_range":
                no_trade_reason = "Trading is blocked until 09:45 ET."
            elif clock_state.flatten_required:
                no_trade_reason = "No new trade is allowed near the close."
            else:
                no_trade_reason = f"Trading is not allowed in the {clock_state.session_window} window."

        return ORBState(
            symbol=symbol.upper(),
            timestamp=reference_time.astimezone(self.market_clock.exchange_tz).isoformat(),
            exchange_time=clock_state.exchange_time.isoformat(),
            range_start=clock_state.market_open_time.replace(
                hour=self.settings.session.orb_start.hour,
                minute=self.settings.session.orb_start.minute,
                second=0,
                microsecond=0,
            ).isoformat(),
            range_end=clock_state.market_open_time.replace(
                hour=self.settings.session.orb_end.hour,
                minute=self.settings.session.orb_end.minute,
                second=0,
                microsecond=0,
            ).isoformat(),
            range_high=opening_range.high,
            range_low=opening_range.low,
            range_mid=opening_range.mid,
            range_width=opening_range.width,
            range_complete=opening_range.complete,
            breakout_direction=breakout_direction,
            breakout_price=current_price if breakout_direction else None,
            breakout_distance=breakout_distance,
            session_window=clock_state.session_window,
            trading_allowed=trading_allowed and not clock_state.flatten_required,
            flatten_required=clock_state.flatten_required,
            time_to_close_minutes=clock_state.minutes_to_close,
            candidate_reason=candidate_reason,
            no_trade_reason=no_trade_reason,
        )

    def _compute_opening_range(
        self,
        bars: pd.DataFrame,
        clock_state: MarketClockState,
    ) -> OpeningRange:
        if bars.empty:
            return OpeningRange(None, None, None, None, False)

        frame = bars.copy()
        if "timestamp" not in frame.columns:
            raise ValueError("Historical bar frame must include a timestamp column.")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
            self.market_clock.exchange_tz
        )
        start_time = self.settings.session.orb_start
        end_time = self.settings.session.orb_end
        session_date = clock_state.exchange_time.date()
        todays_bars = frame[frame["timestamp"].dt.date == session_date]
        range_bars = todays_bars[
            (todays_bars["timestamp"].dt.time >= start_time)
            & (todays_bars["timestamp"].dt.time < end_time)
        ]
        if range_bars.empty or clock_state.exchange_time.timetz().replace(tzinfo=None) < end_time:
            return OpeningRange(None, None, None, None, False)

        high = float(range_bars["high"].max())
        low = float(range_bars["low"].min())
        width = high - low
        mid = low + width / 2 if width is not None else None
        return OpeningRange(high, low, mid, width, True)
