# MicroAlpha IBKR Paper Trading Foundation

## Overview

This repository is a paper-only trading system for Interactive Brokers through IB Gateway. The codebase combines four pieces:

1. A broker CLI for connectivity, market data, account state, manual paper orders, cancellation, and manual closes.
2. A session engine that runs one decision cycle for a single symbol, builds ORB and microstructure features, evaluates risk, stores the decision, and can optionally submit a paper trade.
3. A small research stack with a baseline sklearn model and a lightweight PyTorch sequence model.
4. A local Streamlit console for inspecting market state, models, decisions, trades, and manual controls.

The default flow is still safety-first: paper only, dry-run by default, and explicit confirmation required before any real paper submission.

## What The Project Does

Today the project can:

- connect to IB Gateway paper (`127.0.0.1:4002` by default)
- fetch server time, account summary, positions, open orders, market snapshots, and intraday bars
- compute an opening-range breakout state for the configured symbol
- derive microstructure features and persist them in SQLite
- run baseline and deep model inference when artifacts are registered
- generate a final decision with risk checks and an explanation
- optionally submit paper orders when every safety gate is open
- track trade lifecycle events and execution events
- log execution audit rows to CSV
- expose the workflow in a local Streamlit UI

It does not do live trading. It is still built around deliberate, single-cycle paper execution.

## Project Layout

- `app.py`: main CLI entrypoint
- `broker/`: IBKR connectivity, contracts, and order builders
- `engine/`: runtime wiring, market clock, and session cycle orchestration
- `strategy/`: ORB logic, signal assembly, and decision explanation
- `features/`: feature engineering from market and ORB context
- `models/`: training, inference, and model registry
- `storage/` and `reporting/`: SQLite stores, CSV audit logging, and summaries
- `ui/`: Streamlit application
- `data/`: schemas, loaders, feature store, and sample dataset

## Safety Notes

- Paper trading only. IB Gateway paper must already be running locally.
- `DRY_RUN=true` is the default.
- `SAFE_TO_TRADE=false` is the default.
- Session-triggered execution also requires `ALLOW_SESSION_EXECUTION=true`.
- Manual test orders require `--confirm-paper` in addition to the environment flags.
- Use tiny size first, such as `1` share.
- If IB Gateway API is in read-only mode, order placement, cancellation, and some order queries will fail.
- If `pandas-market-calendars` is not available, the market clock falls back to regular-hours timing without holiday modeling.

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

The defaults are intentionally safe. The most important values are:

```dotenv
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=1
IB_SYMBOL=SPY
DRY_RUN=true
SAFE_TO_TRADE=false
ALLOW_SESSION_EXECUTION=false
DEFAULT_ORDER_QUANTITY=1
```

## Execution Commands

### Start with help

```bash
python app.py --help
```

### Safest first connection check

```bash
python app.py check-connection
```

### Read-only broker commands

```bash
python app.py server-time
python app.py account-summary
python app.py positions
python app.py snapshot
python app.py snapshot --symbol SPY
python app.py open-orders
```

### One safe session cycle

```bash
python app.py session-cycle
```

Alias:

```bash
python app.py session
```

### Request paper execution during a session cycle

This only submits if all execution gates are open:

- `SAFE_TO_TRADE=true`
- `DRY_RUN=false`
- `ALLOW_SESSION_EXECUTION=true`
- explicit `--paper`

```bash
python app.py session-cycle --paper
```

### Inspect the last stored decision

```bash
python app.py explain-latest-decision
```

Alias:

```bash
python app.py latest-decision
```

### Intentional tiny paper test order

Safe preview:

```bash
python app.py place-test-order --action BUY --quantity 1
```

Real paper request:

```bash
python app.py place-test-order --action BUY --quantity 1 --confirm-paper
```

Alias:

```bash
python app.py test-order --action BUY --quantity 1
```

### Legacy manual order commands

```bash
python app.py market-order --action BUY --quantity 1
python app.py limit-order --action BUY --quantity 1 --limit-price 500
python app.py bracket-order --action BUY --quantity 1 --entry-limit 500 --take-profit 505 --stop-loss 495
python app.py cancel-order --order-id 123
python app.py close-position
python app.py close-position --symbol SPY
```

### Model and research commands

```bash
python app.py train-baseline
python app.py train-baseline --data-path data/sample/spy_microstructure_sample.csv
python app.py train-deep --epochs 6
python app.py list-models
python app.py list-models --model-type baseline
python app.py set-active-model --model-type baseline --artifact-id <artifact-id>
```

Alias:

```bash
python app.py models
```

### Launch the local UI

```bash
python app.py launch-ui
```

Alias:

```bash
python app.py ui
```

Direct Streamlit launch also works:

```bash
python -m streamlit run ui/streamlit_app.py
```

## Runtime Outputs

- app logs: `logs/ibkr_mvp.log`
- execution audit CSV: `logs/executions.csv`
- SQLite runtime store: `runtime/microalpha.db`
- model artifacts: `models/artifacts/`

## Tests

Run the local tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests
```
