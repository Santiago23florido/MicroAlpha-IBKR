from __future__ import annotations

from typing import Any


def build_prediction_reasons(prediction: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not prediction.get("valid", True):
        reasons.extend(prediction.get("reasons", []))
        return reasons

    target_mode = prediction.get("target_mode")
    probability = prediction.get("probability")
    predicted_return_bps = prediction.get("predicted_return_bps")
    quantiles = prediction.get("predicted_quantiles") or {}
    action_bias = prediction.get("action_bias")

    reasons.append(f"model={prediction.get('model_name')} target_mode={target_mode}")
    if action_bias:
        reasons.append(f"action_bias={action_bias}")
    if probability is not None:
        reasons.append(f"probability={float(probability):.4f}")
    if predicted_return_bps is not None:
        reasons.append(f"predicted_return_bps={float(predicted_return_bps):.4f}")
    if quantiles:
        ordered = ", ".join(f"{key}={value:.4f}" for key, value in sorted(quantiles.items()))
        reasons.append(f"quantiles[{ordered}]")
    return reasons


def build_threshold_reason(name: str, passed: bool, detail: str) -> str:
    prefix = "passed" if passed else "blocked"
    return f"{prefix}:{name}:{detail}"


def build_risk_reasons(failures: list[str]) -> list[str]:
    return [f"risk:{failure}" for failure in failures]
