from __future__ import annotations

from risk.risk_manager import ExecutionRequest, RiskManager


def build_risk_manager(*, safe_to_trade: bool = False, dry_run: bool = True) -> RiskManager:
    return RiskManager(
        safe_to_trade=safe_to_trade,
        dry_run=dry_run,
        supported_symbols=("SPY",),
    )


def test_risk_blocks_default_execution_mode() -> None:
    risk_manager = build_risk_manager()
    decision = risk_manager.evaluate_execution_request(
        ExecutionRequest(
            symbol="SPY",
            action="BUY",
            quantity=1,
            order_type="market",
            explicit_command=True,
        )
    )

    assert decision.approved is False
    assert "SAFE_TO_TRADE is false" in decision.reason


def test_risk_rejects_unsupported_symbol() -> None:
    risk_manager = build_risk_manager(safe_to_trade=True, dry_run=False)
    decision = risk_manager.evaluate_execution_request(
        ExecutionRequest(
            symbol="AAPL",
            action="BUY",
            quantity=1,
            order_type="market",
            explicit_command=True,
        )
    )

    assert decision.approved is False
    assert "not enabled" in decision.reason


def test_risk_allows_dry_run_preview_when_trade_flag_is_enabled() -> None:
    risk_manager = build_risk_manager(safe_to_trade=True, dry_run=True)
    decision = risk_manager.evaluate_execution_request(
        ExecutionRequest(
            symbol="SPY",
            action="BUY",
            quantity=1,
            order_type="limit",
            explicit_command=True,
            limit_price=500.0,
        )
    )

    assert decision.approved is True
    assert decision.submit_to_broker is False
    assert "Dry-run is enabled" in decision.reason


def test_risk_rejects_invalid_bracket_prices() -> None:
    risk_manager = build_risk_manager(safe_to_trade=True, dry_run=False)
    decision = risk_manager.evaluate_execution_request(
        ExecutionRequest(
            symbol="SPY",
            action="BUY",
            quantity=1,
            order_type="bracket",
            explicit_command=True,
            limit_price=100.0,
            take_profit_price=99.0,
            stop_loss_price=98.0,
        )
    )

    assert decision.approved is False
    assert "bracket prices are inconsistent" in decision.reason
