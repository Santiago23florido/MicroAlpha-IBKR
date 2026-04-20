# Paper Session Start

1. Run `python app.py preflight-check`.
2. If status is `ok`, run `python app.py run-paper-validation-session`.
3. Review `data/reports/sessions/<session_id>/session_summary.json`.
4. Review `alerts_summary.csv` and `reconciliation_summary.json`.
5. Only continue if readiness is not `NOT_READY`.
