# Preflight Checklist

1. Confirm `ACTIVE_EXECUTION_BACKEND=ibkr_paper`.
2. Confirm `BROKER_MODE=paper`.
3. Confirm `SAFE_TO_TRADE=true`.
4. Confirm `ALLOW_SESSION_EXECUTION=true`.
5. Run `python app.py broker-healthcheck`.
6. Run `python app.py show-active-model`.
7. Run `python app.py preflight-check`.
8. If any critical check fails, do not start a paper validation session.
