from __future__ import annotations

from pathlib import Path
from typing import Any

from config import Settings
from config.phase10_11 import load_phase10_11_config


RUNBOOK_CONTENT = {
    "preflight_checklist.md": """# Preflight Checklist

1. Confirm `ACTIVE_EXECUTION_BACKEND=ibkr_paper`.
2. Confirm `BROKER_MODE=paper`.
3. Confirm `SAFE_TO_TRADE=true`.
4. Confirm `ALLOW_SESSION_EXECUTION=true`.
5. Run `python app.py broker-healthcheck`.
6. Run `python app.py show-active-model`.
7. Run `python app.py preflight-check`.
8. If any critical check fails, do not start a paper validation session.
""",
    "paper_session_start.md": """# Paper Session Start

1. Run `python app.py preflight-check`.
2. If status is `ok`, run `python app.py run-paper-validation-session`.
3. Review `data/reports/sessions/<session_id>/session_summary.json`.
4. Review `alerts_summary.csv` and `reconciliation_summary.json`.
5. Only continue if readiness is not `NOT_READY`.
""",
    "reconciliation_review.md": """# Reconciliation Review

1. Run `python app.py reconcile-broker-state`.
2. Review `orders_*.csv`, `fills_*.csv`, and `positions_*.csv`.
3. If any critical mismatch exists, stop further paper validation.
4. Resolve quantity or broker-order mapping mismatches before the next session.
""",
    "incident_response.md": """# Incident Response

1. Run `python app.py list-incidents`.
2. Identify the latest critical incident.
3. Review the related alerts and recovery events.
4. If the incident is broker connectivity or reconciliation related, do not resume automatically.
5. Only restart after the root cause is understood and documented.
""",
    "recovery_steps.md": """# Recovery Steps

1. Check `python app.py execution-status --limit 10`.
2. If open orders remain, do not auto-resume.
3. Run `python app.py broker-healthcheck`.
4. Run `python app.py reconcile-broker-state`.
5. If the state is clean, restart with `python app.py full-paper-validation-cycle`.
""",
    "not_ready_actions.md": """# NOT_READY Actions

1. Review `readiness_report.json`.
2. Review critical alerts and unresolved incidents.
3. Review reconciliation mismatches and drawdown/drift warnings.
4. Do not continue automated paper validation until the blocking reasons are resolved.
5. Re-run `python app.py generate-readiness-report` after remediation.
""",
}


def generate_runbooks(settings: Settings) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    runbooks_dir = Path(phase10_11.report_paths.runbooks_dir)
    runbooks_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for filename, content in RUNBOOK_CONTENT.items():
        target = runbooks_dir / filename
        target.write_text(content, encoding="utf-8")
        paths[filename] = str(target)
    return {
        "status": "ok",
        "runbooks_dir": str(runbooks_dir),
        "files": paths,
    }
