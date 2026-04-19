from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from config import load_settings
from config.phase6 import load_active_model_selection
from config.phase7 import load_phase7_config
from engine import phase7 as phase7_engine
from engine.phase7 import broker_healthcheck, run_paper_session_real
from execution import ExecutionJournal, IBKRPaperExecutionBackend, ModelTrace, Order, OrderAction, OrderManager, OrderStatus, OrderType, PositionManager
from tests.test_phase6_operations import PROJECT_ROOT, create_mock_feature_store, create_mock_phase5_artifact


def build_phase9_test_settings(tmp_path: Path, *, extra_env: list[str] | None = None):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "settings.yaml",
        "risk.yaml",
        "symbols.yaml",
        "deployment.yaml",
        "feature_sets.yaml",
        "modeling.yaml",
        "phase6.yaml",
        "phase7.yaml",
        "phase8.yaml",
    ):
        shutil.copy(PROJECT_ROOT / "config" / name, config_dir / name)
    env_lines = [
        "APP_ENV=development",
        "SUPPORTED_SYMBOLS=SPY",
        "DRY_RUN=false",
        "SAFE_TO_TRADE=true",
        "ALLOW_SESSION_EXECUTION=true",
        "ACTIVE_EXECUTION_BACKEND=ibkr_paper",
        "BROKER_MODE=paper",
        "IBKR_PAPER_CLIENT_ID=901",
    ]
    env_lines.extend(extra_env or [])
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(env_lines), encoding="utf-8")
    return load_settings(env_file=env_file, config_dir=config_dir, environment="development")


class FakeIBClient:
    def __init__(self, *, status: str = "Filled") -> None:
        self._connected = False
        self.status = status
        self.order_id = 9001
        self.perm_id = 7001

    def connect(self, timeout: float | None = None) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_server_time(self) -> dict[str, str | int]:
        return {"epoch": 1776606000, "iso_utc": "2026-04-19T13:40:00+00:00"}

    def get_account_summary(self) -> list[dict[str, str]]:
        return [{"account": "DU123456", "tag": "AccountType", "value": "INDIVIDUAL", "currency": "USD"}]

    def get_open_orders(self) -> list[dict[str, object]]:
        return []

    def get_positions(self) -> list[dict[str, object]]:
        return [{"account": "DU123456", "symbol": "SPY", "position": 0.0, "avgCost": 0.0}]

    def request_recent_executions(self) -> list[dict[str, object]]:
        return []

    def submit_market_order(self, **kwargs):
        return self._snapshot(kwargs["action"], kwargs["quantity"])

    def submit_limit_order(self, **kwargs):
        return self._snapshot(kwargs["action"], kwargs["quantity"], limit_price=kwargs.get("limit_price"))

    def submit_stop_order(self, **kwargs):
        return self._snapshot(kwargs["action"], kwargs["quantity"], stop_price=kwargs.get("stop_price"))

    def submit_stop_limit_order(self, **kwargs):
        return self._snapshot(
            kwargs["action"],
            kwargs["quantity"],
            limit_price=kwargs.get("limit_price"),
            stop_price=kwargs.get("stop_price"),
        )

    def cancel_order(self, order_id: int) -> dict[str, object]:
        payload = self._snapshot("BUY", 1)
        payload["order_id"] = order_id
        payload["status"] = "Cancelled"
        payload["final_at_utc"] = "2026-04-19T13:40:03+00:00"
        payload["executions"] = []
        return payload

    def get_order_status(self, order_id: int) -> dict[str, object]:
        payload = self._snapshot("BUY", 1)
        payload["order_id"] = order_id
        return payload

    def _snapshot(
        self,
        action: str,
        quantity: int,
        *,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> dict[str, object]:
        executions = []
        filled_quantity = 0
        remaining_quantity = quantity
        average_fill_price = None
        final_at_utc = None
        if self.status == "Filled":
            filled_quantity = quantity
            remaining_quantity = 0
            average_fill_price = 100.02
            final_at_utc = "2026-04-19T13:40:02+00:00"
            executions = [
                {
                    "execution_id": "EX123",
                    "order_id": self.order_id,
                    "broker_perm_id": self.perm_id,
                    "shares": quantity,
                    "price": 100.02,
                    "exchange": "SMART",
                    "execution_time": "20260419  13:40:02",
                    "timestamp_utc": "2026-04-19T13:40:02+00:00",
                    "commission_report": {"commission": 0.27, "currency": "USD"},
                }
            ]
        return {
            "order_id": self.order_id,
            "broker_perm_id": self.perm_id,
            "status": self.status,
            "filled_quantity": filled_quantity,
            "remaining_quantity": remaining_quantity,
            "average_fill_price": average_fill_price,
            "last_fill_price": average_fill_price,
            "acknowledged_at_utc": "2026-04-19T13:40:01+00:00",
            "updated_at_utc": "2026-04-19T13:40:01+00:00",
            "final_at_utc": final_at_utc,
            "status_history": [
                {"timestamp_utc": "2026-04-19T13:40:01+00:00", "status": "Submitted", "filled_quantity": 0}
            ],
            "executions": executions,
            "message": None,
            "action": action,
            "quantity": quantity,
            "limit_price": limit_price,
            "stop_price": stop_price,
        }


def selection_to_model_trace(selection) -> ModelTrace:
    return ModelTrace(
        model_name=selection.model_name,
        model_type=selection.model_type,
        run_id=selection.run_id,
        feature_set_name=selection.feature_set_name,
        target_mode=selection.target_mode,
        artifact_dir=selection.artifact_dir,
        selection_reason=selection.selection_reason,
        source_leaderboard=selection.source_leaderboard,
        updated_at_utc=selection.updated_at_utc,
    )


def test_ibkr_paper_backend_healthcheck_and_submission(tmp_path: Path) -> None:
    settings = build_phase9_test_settings(tmp_path)
    phase7 = load_phase7_config(settings)
    backend = IBKRPaperExecutionBackend(phase7, settings=settings, client=FakeIBClient())

    health = backend.healthcheck()
    order = Order(
        order_id="ord_real_001",
        symbol="SPY",
        action=OrderAction.BUY,
        quantity=1,
        order_type=OrderType.MARKET,
        status=OrderStatus.CREATED,
        created_at="2026-04-19T13:40:00+00:00",
        updated_at="2026-04-19T13:40:00+00:00",
        source_model_name="logistic_regression",
        source_decision_id="dec_real_001",
        backend_name="ibkr_paper",
        metadata={
            "decision_generated_at_utc": "2026-04-19T13:40:00+00:00",
            "decision_action": "LONG",
            "decision_snapshot": {
                "action": "LONG",
                "size_suggestion": 1,
                "expected_cost_bps": 1.8,
                "blocked_by_risk": False,
                "risk_checks": {"spread_ok": True},
            },
        },
    )

    result = backend.submit_order(order, market_data={"ask": 100.0, "bid": 99.99, "last": 100.0})

    assert health["status"] == "ok"
    assert result.accepted is True
    assert result.broker_order_id == 9001
    assert result.fills[0].execution_id == "EX123"
    assert result.metadata["latency_ms"]["submit_to_ack_ms"] is not None
    assert result.metadata["decision_vs_execution"]["filled_size"] == 1


def test_ibkr_paper_backend_rejects_live_mode(tmp_path: Path) -> None:
    settings = build_phase9_test_settings(tmp_path, extra_env=["BROKER_MODE=live"])
    phase7 = load_phase7_config(settings)
    backend = IBKRPaperExecutionBackend(phase7, settings=settings, client=FakeIBClient())

    with pytest.raises(ValueError, match="broker_mode"):
        backend.healthcheck()


def test_order_manager_preserves_broker_mapping_and_reconciliation(tmp_path: Path) -> None:
    settings = build_phase9_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    phase7 = load_phase7_config(settings)
    selection = load_active_model_selection(settings)
    backend = IBKRPaperExecutionBackend(phase7, settings=settings, client=FakeIBClient())
    journal = ExecutionJournal(phase7.logging)
    positions = PositionManager(initial_cash=phase7.session.initial_cash)
    manager = OrderManager(
        phase7,
        backend=backend,
        journal=journal,
        position_manager=positions,
    )

    result = manager.process_decision(
        {
            "symbol": "SPY",
            "action": "LONG",
            "size_suggestion": 1,
            "timestamp": "2026-04-19T13:40:00+00:00",
            "blocked_by_risk": False,
            "risk_checks": {"spread_ok": True, "model_output_valid": True},
            "reasons": ["passed:spread_limit"],
        },
        model_trace=selection_to_model_trace(selection),
        decision_id="dec_real_order_manager",
        market_data={"ask": 100.0, "bid": 99.99, "last": 100.0},
    )

    assert result.accepted is True
    assert result.order.status.value == "FILLED"
    assert result.order.broker_order_id == 9001
    assert result.order.backend_name == "ibkr_paper"
    assert journal.recent_reconciliation(5)


def test_run_paper_session_real_uses_ibkr_backend_and_writes_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = build_phase9_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)

    def fake_build_execution_backend(config, **kwargs):
        return IBKRPaperExecutionBackend(config, settings=settings, client=FakeIBClient())

    monkeypatch.setattr(phase7_engine, "build_execution_backend", fake_build_execution_backend)

    summary = run_paper_session_real(settings, symbols=["SPY"], latest_per_symbol=1)
    health = broker_healthcheck(settings)

    assert summary["status"] == "ok"
    assert summary["backend_name"] == "ibkr_paper"
    assert Path(summary["summary_path"]).exists()
    assert health["status"] == "ok"
