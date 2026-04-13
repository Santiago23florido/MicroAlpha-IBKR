from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

try:
    import pandas_market_calendars as mcal
except Exception:  # pragma: no cover - optional runtime dependency
    mcal = None

from config import SessionSettings


@dataclass(frozen=True)
class MarketClockState:
    exchange_time: datetime
    market_open_time: datetime
    market_close_time: datetime
    is_market_open: bool
    is_primary_window: bool
    is_secondary_window: bool
    session_window: str
    minutes_to_close: float
    flatten_required: bool
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "exchange_time": self.exchange_time.isoformat(),
            "market_open_time": self.market_open_time.isoformat(),
            "market_close_time": self.market_close_time.isoformat(),
            "is_market_open": self.is_market_open,
            "is_primary_window": self.is_primary_window,
            "is_secondary_window": self.is_secondary_window,
            "session_window": self.session_window,
            "minutes_to_close": self.minutes_to_close,
            "flatten_required": self.flatten_required,
            "note": self.note,
        }


class MarketClock:
    def __init__(self, session_settings: SessionSettings) -> None:
        self.settings = session_settings
        self.exchange_tz = ZoneInfo(session_settings.timezone)
        self._calendar = mcal.get_calendar("XNYS") if mcal is not None else None

    def now_exchange(self) -> datetime:
        return datetime.now(timezone.utc).astimezone(self.exchange_tz)

    def to_exchange_time(self, value: datetime | None) -> datetime:
        if value is None:
            return self.now_exchange()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(self.exchange_tz)

    def get_market_state(self, reference: datetime | None = None) -> MarketClockState:
        exchange_time = self.to_exchange_time(reference)
        session_open, session_close, note = self._session_bounds(exchange_time.date())
        is_market_open = session_open <= exchange_time < session_close
        is_primary_window = self._is_between(
            exchange_time.timetz().replace(tzinfo=None),
            self.settings.orb_end,
            self.settings.primary_session_end,
        )
        is_secondary_window = self.settings.enable_secondary_session and self._is_between(
            exchange_time.timetz().replace(tzinfo=None),
            self.settings.secondary_session_start,
            self.settings.secondary_session_end,
        )
        if exchange_time < session_open:
            session_window = "pre_market"
        elif exchange_time.timetz().replace(tzinfo=None) < self.settings.orb_end:
            session_window = "opening_range"
        elif is_primary_window:
            session_window = "primary"
        elif is_secondary_window:
            session_window = "secondary"
        elif exchange_time >= session_close:
            session_window = "closed"
        else:
            session_window = "between_windows"
        minutes_to_close = max((session_close - exchange_time).total_seconds() / 60.0, 0.0)
        flatten_required = minutes_to_close <= self.settings.flatten_before_close_minutes
        return MarketClockState(
            exchange_time=exchange_time,
            market_open_time=session_open,
            market_close_time=session_close,
            is_market_open=is_market_open,
            is_primary_window=is_primary_window,
            is_secondary_window=is_secondary_window,
            session_window=session_window,
            minutes_to_close=minutes_to_close,
            flatten_required=flatten_required,
            note=note,
        )

    def _session_bounds(self, session_date: date) -> tuple[datetime, datetime, str]:
        if self._calendar is not None:
            schedule = self._calendar.schedule(
                start_date=session_date.isoformat(),
                end_date=session_date.isoformat(),
            )
            if not schedule.empty:
                open_time = schedule.iloc[0]["market_open"].to_pydatetime().astimezone(self.exchange_tz)
                close_time = schedule.iloc[0]["market_close"].to_pydatetime().astimezone(self.exchange_tz)
                return open_time, close_time, "NYSE calendar schedule."

        open_time = datetime.combine(session_date, self.settings.regular_market_open, tzinfo=self.exchange_tz)
        close_time = datetime.combine(session_date, self.settings.regular_market_close, tzinfo=self.exchange_tz)
        return open_time, close_time, "Fallback regular-hours clock without exchange holiday modeling."

    @staticmethod
    def _is_between(current: time, start: time, end: time) -> bool:
        return start <= current <= end
