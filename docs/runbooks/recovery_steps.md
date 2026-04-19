# Recovery Steps

1. Check `python app.py execution-status --limit 10`.
2. If open orders remain, do not auto-resume.
3. Run `python app.py broker-healthcheck`.
4. Run `python app.py reconcile-broker-state`.
5. If the state is clean, restart with `python app.py full-paper-validation-cycle`.
