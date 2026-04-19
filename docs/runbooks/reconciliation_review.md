# Reconciliation Review

1. Run `python app.py reconcile-broker-state`.
2. Review `orders_*.csv`, `fills_*.csv`, and `positions_*.csv`.
3. If any critical mismatch exists, stop further paper validation.
4. Resolve quantity or broker-order mapping mismatches before the next session.
