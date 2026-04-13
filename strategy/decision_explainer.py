from __future__ import annotations

from typing import Any

from data.schemas import ModelPrediction, ORBState, RiskCheckResult


class DecisionExplainer:
    def build(
        self,
        *,
        orb_state: ORBState,
        baseline_prediction: ModelPrediction,
        deep_prediction: ModelPrediction,
        threshold_checks: dict[str, bool],
        risk_result: RiskCheckResult,
        execution_gates: dict[str, bool],
        final_action: str,
        expected_edge: float | None,
        estimated_cost: float | None,
    ) -> tuple[str, dict[str, Any]]:
        reasons: list[str] = []
        reasons.append(orb_state.candidate_reason)
        if orb_state.no_trade_reason:
            reasons.append(orb_state.no_trade_reason)

        if baseline_prediction.eligible:
            reasons.append(
                f"Baseline model {baseline_prediction.model_name} direction={baseline_prediction.direction or 'flat'} "
                f"prob_up={baseline_prediction.probability_up:.3f} prob_down={baseline_prediction.probability_down:.3f}."
            )
        else:
            reasons.append("; ".join(baseline_prediction.reasons))

        if deep_prediction.eligible:
            reasons.append(
                f"Deep model {deep_prediction.model_name} direction={deep_prediction.direction or 'flat'} "
                f"prob_up={deep_prediction.probability_up:.3f} prob_down={deep_prediction.probability_down:.3f}."
            )
        else:
            reasons.append("; ".join(deep_prediction.reasons))

        if expected_edge is not None and estimated_cost is not None:
            if threshold_checks.get("expected_edge_exceeds_cost", False):
                reasons.append(
                    f"Estimated edge {expected_edge:.2f} bps exceeds cost hurdle {estimated_cost:.2f} bps."
                )
            else:
                reasons.append(
                    f"Estimated edge {expected_edge:.2f} bps does not exceed cost hurdle {estimated_cost:.2f} bps."
                )

        if not risk_result.passed:
            reasons.append(risk_result.summary)

        blocked_gates = [name for name, passed in execution_gates.items() if not passed]
        if blocked_gates:
            reasons.append(
                "Paper execution remains blocked by: " + ", ".join(name.replace("_", " ") for name in blocked_gates) + "."
            )

        reasons.append(f"Final action: {final_action}.")

        structured = {
            "orb": orb_state.to_dict(),
            "baseline_model": baseline_prediction.to_dict(),
            "deep_model": deep_prediction.to_dict(),
            "threshold_checks": threshold_checks,
            "risk": risk_result.to_dict(),
            "execution_gates": execution_gates,
        }
        return " ".join(reason for reason in reasons if reason), structured
