# MicroAlpha-IBKR Phase 2 Collector Foundation

## Purpose of Phase 2

Phase 1 left the repository organized for a two-machine setup:

- `PC1`: development, research, training, backtesting, validation
- `PC2`: deployment, market data collection, operational monitoring, later inference

Phase 2 implements the first operational component for `PC2`: a robust market data collector for IBKR.

This phase does **not** implement autonomous trading, strategy execution, final paper trading, advanced backtesting, news, RL, or online inference. It focuses on a clean and extensible data collection node.

## What Is New in Phase 2

The repository now includes:

- a dedicated collector client layer in [ingestion/ibkr_client.py](/home/santiago/ibkr_invest/ingestion/ibkr_client.py)
- normalized market data records in [ingestion/market_data.py](/home/santiago/ibkr_invest/ingestion/market_data.py)
- parquet persistence with batching in [ingestion/persistence.py](/home/santiago/ibkr_invest/ingestion/persistence.py)
- a reconnecting polling collector loop in [ingestion/collector.py](/home/santiago/ibkr_invest/ingestion/collector.py)
- stronger collector health reporting in [monitoring/healthcheck.py](/home/santiago/ibkr_invest/monitoring/healthcheck.py)
- collector-specific config values in [config/settings.yaml](/home/santiago/ibkr_invest/config/settings.yaml)

## Architecture

### PC1

Use `development` mode for:

- training
- backtesting
- session validation
- dashboard inspection

### PC2

Use `deploy` mode for:

- IBKR connectivity
- continuous or bounded market data polling
- parquet persistence under `data/raw/market`
- operational health checks

## Collector Design

The collector intentionally uses polling, not a second low-level streaming stack. That choice is deliberate:

- the repo already had a working IBKR client for snapshots and historical fallback
- polling is simpler to deploy and observe on PC2
- it avoids introducing a parallel IBKR implementation before phase 3

Current flow:

1. load config from `.env` + `config/*.yaml`
2. build collector logger
3. validate output path
4. connect to IBKR using a dedicated collector `clientId`
5. poll snapshots for configured symbols
6. normalize records to one schema
7. buffer in memory
8. flush parquet batches by date and symbol
9. log health, errors and reconnect attempts
10. flush and disconnect cleanly on exit

## Project Structure

```text
MicroAlpha-IBKR/
├── app.py
├── config/
├── data/
│   ├── raw/
│   │   └── market/
│   ├── processed/
│   ├── features/
│   ├── models/
│   ├── logs/
│   └── reports/
├── ingestion/
│   ├── ibkr_client.py
│   ├── market_data.py
│   ├── persistence.py
│   └── collector.py
├── monitoring/
│   ├── logging.py
│   ├── healthcheck.py
│   └── sync.py
├── scripts/
│   ├── run_collector.py
│   ├── healthcheck.py
│   ├── run_session.py
│   ├── train_model.py
│   └── sync_data.py
└── tests/
```

Existing working modules from earlier phases are still present:

- `broker/`
- `engine/`
- `strategy/`
- `risk/`
- `models/`
- `ui/`
- `storage/`

## Configuration

Configuration is merged from:

- `.env`
- [config/settings.yaml](/home/santiago/ibkr_invest/config/settings.yaml)
- [config/risk.yaml](/home/santiago/ibkr_invest/config/risk.yaml)
- [config/symbols.yaml](/home/santiago/ibkr_invest/config/symbols.yaml)
- [config/deployment.yaml](/home/santiago/ibkr_invest/config/deployment.yaml)

### Environment modes

- `development`
- `deploy`

### Collector-specific settings

- `IB_COLLECTOR_CLIENT_ID`
- `COLLECTOR_MODE`
- `COLLECTOR_POLL_INTERVAL_SECONDS`
- `COLLECTOR_FLUSH_INTERVAL_SECONDS`
- `COLLECTOR_BATCH_SIZE`
- `COLLECTOR_RECONNECT_DELAY_SECONDS`
- `COLLECTOR_MAX_RECONNECT_ATTEMPTS`
- `COLLECTOR_HEALTH_LOG_INTERVAL_SECONDS`

Inspect the effective merged config with:

```bash
python app.py show-config
python app.py --environment deploy show-config
```

## IBKR Setup for PC2

Before running the collector on the deployment machine:

1. open `IB Gateway Paper` or `TWS Paper`
2. enable socket/API access
3. confirm the correct port:
   - `IB Gateway Paper`: usually `4002`
   - `TWS Paper`: usually `7497`
4. keep `IB_COLLECTOR_CLIENT_ID` different from:
   - `IB_CLIENT_ID`
   - `IB_UI_CLIENT_ID`

Recommended `.env` values for PC2:

```dotenv
APP_ENV=deploy
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=1
IB_UI_CLIENT_ID=101
IB_COLLECTOR_CLIENT_ID=201
SUPPORTED_SYMBOLS=SPY
COLLECTOR_ENABLED=true
COLLECTOR_POLL_INTERVAL_SECONDS=5
COLLECTOR_FLUSH_INTERVAL_SECONDS=20
COLLECTOR_BATCH_SIZE=50
COLLECTOR_RECONNECT_DELAY_SECONDS=10
COLLECTOR_MAX_RECONNECT_ATTEMPTS=5
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Main Commands

### Collector

One polling cycle:

```bash
python app.py collect --once
```

Continuous run:

```bash
python app.py --environment deploy collect
```

Bounded run:

```bash
python app.py --environment deploy collect --max-cycles 12
python app.py --environment deploy collect --max-runtime-seconds 300
```

Override polling and flush parameters:

```bash
python app.py --environment deploy collect \
  --symbols SPY QQQ \
  --poll-interval 3 \
  --flush-interval 15 \
  --batch-size 25
```

### Healthcheck

Config and writable paths only:

```bash
python app.py healthcheck --skip-broker
```

Full collector-oriented broker check:

```bash
python app.py --environment deploy healthcheck
```

### Script entrypoints

```bash
python scripts/run_collector.py --environment deploy --once
python scripts/healthcheck.py --environment deploy --skip-broker
```

### Other preserved workflows

```bash
python app.py run-session
python app.py train --model-type baseline
python app.py backtest
python app.py dashboard
```

## Output Data Layout

Collected market data is written under:

```text
data/raw/market/YYYY-MM-DD/SYMBOL/collector_*.parquet
```

This layout is intentionally partitioned by day and symbol instead of rewriting a single large file. It is safer for long-running collection and easier to sync from PC2 later.

### Current normalized columns

- `timestamp`
- `symbol`
- `last_price`
- `bid`
- `ask`
- `spread`
- `bid_size`
- `ask_size`
- `last_size`
- `volume`
- `event_type`
- `source`
- `session_window`
- `is_market_open`
- `exchange_time`
- `collected_at`

## Logging

Logs are written through the shared logger in [monitoring/logging.py](/home/santiago/ibkr_invest/monitoring/logging.py).

The collector logs at least:

- startup
- IBKR connection attempts
- reconnect attempts
- polling health heartbeat
- parquet flush counts
- shutdown

Default log file:

```text
data/logs/microalpha.log
```

## Healthcheck Coverage

The healthcheck now validates:

- config loading
- environment mode
- collector symbols
- collector output path
- output path writability
- IBKR reachability through the collector client
- dedicated collector `clientId`

## Limitations of Phase 2

Current collector limits are intentional:

- polling snapshots, not a full streaming market data bus
- no execution or order placement
- no online feature generation yet
- no online quality validation beyond basic normalization and persistence
- parquet batches are append-safe by partition, but there is no compaction job yet
- reconnect logic is basic and local-process only

## Phase 3 Targets

Phase 3 can now build directly on this collector foundation:

- online feature generation
- data quality checks and gap detection
- richer event types
- incremental validation of raw market data
- deploy-side inference preparation
- artifact promotion between PC1 and PC2

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests
```

Current local validation completed on this branch:

- CLI help
- healthcheck
- collector unit tests
- parquet persistence tests
- reconnect behavior tests
