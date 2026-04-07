# MicroAlpha IBKR MVP Foundation

## Project Overview

This repository contains a safety-first MVP foundation for Interactive Brokers paper trading through IB Gateway. The current implementation focuses on connectivity, account visibility, one-symbol market data snapshots, and one intentionally manual paper order path that stays blocked by default.

## Current MVP Scope

The first foundation includes:

- connection to IB Gateway paper at `127.0.0.1:4002` with `clientId=1`
- connection health verification
- current IB server time request
- account summary request
- current positions request
- one-symbol market data snapshot
- one explicit manual paper test order command
- environment-based configuration via `.env`
- structured console and file logging
- placeholder strategy and risk modules for future extension

The first foundation does not include:

- automatic trading loops
- strategy execution
- AI decision-making
- live trading
- dashboards
- backtesting
- multi-asset orchestration
- automatic order placement on startup

## Safety Notes

- Paper trading only. This project assumes IB Gateway paper is running locally.
- Dry-run is enabled by default through `DRY_RUN=true`.
- Order submission is blocked by default through `SAFE_TO_TRADE=false`.
- A paper order can only be sent through the explicit `paper-test-order` command.
- Never commit `.env` or any credentials.

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

Review `.env` and keep the defaults unless you intentionally need to change them:

```dotenv
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=1
IB_SYMBOL=SPY
DRY_RUN=true
SAFE_TO_TRADE=false
LOG_LEVEL=INFO
```

## Commands

### Safest first test: connection only

```bash
python app.py check-connection
```

### Request current server time

```bash
python app.py server-time
```

### Request account summary

```bash
python app.py account-summary
```

### Request positions

```bash
python app.py positions
```

### Request one market snapshot

Uses the symbol from `.env`:

```bash
python app.py snapshot
```

Override the symbol manually:

```bash
python app.py snapshot --symbol AAPL
```

### Intentionally run one paper test order command

The default configuration will not submit anything. It will either block the request or keep it as dry-run output until you explicitly change both safety flags.

Dry-run preview only:

```bash
python app.py paper-test-order --action BUY --quantity 1
```

Actual paper submission requires:

1. `SAFE_TO_TRADE=true`
2. `DRY_RUN=false`

Then run:

```bash
python app.py paper-test-order --action BUY --quantity 1
```

## Logging

- Console logs are emitted for every important event.
- File logs are written to `logs/ibkr_mvp.log` by default.

## Test Coverage

Run the connection-focused test suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_connection.py
```
