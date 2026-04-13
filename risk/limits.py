from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StrategyRiskContext:
    symbol: str
    current_date: str
    connection_healthy: bool
    market_is_open: bool
    trading_window_allowed: bool
    flatten_required: bool
    spread_bps: float | None
    open_positions_count: int
    symbol_position: float
    trades_today: int
    daily_realized_pnl: float
    net_liquidation: float | None
    action: str | None
    max_trades_per_day: int
    max_daily_loss_pct: float
    max_open_positions: int
    max_spread_bps: float
    failure_overrides: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionGateContext:
    explicit_session_request: bool
    session_execution_enabled: bool
    safe_to_trade: bool
    dry_run: bool
