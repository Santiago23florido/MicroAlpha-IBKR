# MicroAlpha-IBKR Phase 1 Foundation

## Purpose of Phase 1

This phase reorganizes the repository so the project can evolve into a two-machine workflow without rebuilding the codebase later.

- `PC1`: development, research, training, backtesting, validation
- `PC2`: deployment, scheduled collection, operational inference, monitoring

Phase 1 is intentionally limited. It does **not** implement a final trading bot, advanced training stack, news/LLM integration, RL, or a multi-strategy platform. It leaves a clean foundation for those later phases.

## Current Architecture

The repository is now split around operational concerns:

- `config/`: central YAML configuration for environments, risk, symbols, deployment
- `ingestion/`: collection entrypoints and raw market data capture
- `features/`: feature engineering and preprocessing
- `labels/`: placeholder label pipeline boundary for later phases
- `models/`: training, inference, registry
- `strategy/`: ORB and decision logic
- `execution/`: paper execution helpers
- `risk/`: trading gates and order checks
- `backtest/`: backtest boundary, currently a safe dataset-validation stub
- `monitoring/`: logging, healthcheck, sync planning
- `ui/`: Streamlit dashboard
- `scripts/`: thin executable entrypoints for common workflows
- `data/`: runtime directories for raw/processed/features/models/logs/reports

The existing broker/session/runtime logic was kept and re-used. Phase 1 reorganizes around it instead of discarding working code.

## PC1 vs PC2

### PC1: development / research

- `APP_ENV=development`
- training enabled
- backtest enabled
- collector optional
- safe execution remains disabled by default

Typical commands:

```bash
python app.py train --model-type baseline
python app.py backtest
python app.py run-session
python app.py dashboard
```

### PC2: deploy / operations

- `APP_ENV=deploy`
- collector enabled
- training disabled by default
- backtest disabled by default
- sync and scheduler flags reserved for later deployment automation

Typical commands:

```bash
python app.py collect --environment deploy
python app.py healthcheck --environment deploy
python app.py sync-data --environment deploy --destination-root /path/to/staging
```

## Project Tree

```text
MicroAlpha-IBKR/
├── app.py
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── models/
│   ├── logs/
│   └── reports/
├── ingestion/
├── features/
├── labels/
├── models/
├── strategy/
├── execution/
├── risk/
├── backtest/
├── monitoring/
├── ui/
├── scripts/
└── tests/
```

Also preserved from the original implementation:

- `broker/`: IBKR client and order primitives
- `engine/`: runtime composition and session engine
- `storage/`: SQLite persistence and execution audit storage
- `reporting/`: small trade/performance summaries
- `data/feature_store.py`, `data/live_data.py`, `data/historical_loader.py`, `data/schemas.py`

## Configuration

The project now loads configuration from both `.env` and `config/*.yaml`.

### YAML files

- `config/settings.yaml`
- `config/risk.yaml`
- `config/symbols.yaml`
- `config/deployment.yaml`

### Environment modes

- `development`
- `deploy`

### Important settings

- `environment`
- `data_root`
- `log_level`
- `broker_mode`
- `safe_to_trade`
- `collector_enabled`
- `training_enabled`
- `backtest_enabled`
- `symbols`
- `timezone`
- model, log, report and runtime paths

Inspect the effective merged configuration with:

```bash
python app.py show-config
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## `.env` Setup

Start from `.env.example` and adjust at least:

```dotenv
APP_ENV=development
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=1
IB_UI_CLIENT_ID=101
IB_SYMBOL=SPY
DRY_RUN=true
SAFE_TO_TRADE=false
ALLOW_SESSION_EXECUTION=false
```

Use `IB_UI_CLIENT_ID` different from `IB_CLIENT_ID` so the Streamlit dashboard does not collide with the CLI client session.

## Main Commands

### Base CLI

```bash
python app.py --help
python app.py show-config
python app.py healthcheck
python app.py collect
python app.py train --model-type baseline
python app.py backtest
python app.py run-session
python app.py snapshot
python app.py latest-decision
python app.py models
python app.py dashboard
python app.py sync-data --destination-root /tmp/microalpha-sync
```

### Script entrypoints

```bash
python scripts/run_collector.py
python scripts/run_session.py
python scripts/train_model.py --model-type baseline
python scripts/healthcheck.py
python scripts/sync_data.py --destination-root /tmp/microalpha-sync
```

## What Is Already Operational

- IBKR connectivity checks
- one-session ORB + feature + model + risk cycle
- decision persistence in SQLite
- Streamlit dashboard
- baseline and deep training entrypoints
- market snapshot fallback to historical bars when delayed data is the only source
- raw collector run that persists snapshot and bars under `data/raw/collector/...`
- configuration and path healthcheck

## What Is Still a Placeholder in Phase 1

- `backtest/`: validates datasets and wiring only, no strategy simulation yet
- `labels/`: reserved boundary for later label generation
- `sync-data`: supports local dry-run planning and optional local copying, not remote orchestration
- deploy scheduler automation
- robust long-running collector service for PC2

## Migration Notes

The codebase was not flattened or rewritten from scratch. Useful existing functionality was retained and re-wired:

- the old `config.py` was replaced by the `config/` package and YAML-based settings
- runtime outputs moved conceptually under `data/`
- logging now lives under `monitoring/` with a compatibility wrapper in `storage/logger.py`
- the CLI was simplified around phase-1 workflows instead of low-level order management

This keeps the working broker/session implementation while moving the repo toward a cleaner research/deploy split.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests
```

## Next Phase

Phase 2 can now focus on the actual PC2 deployment work:

- robust collector loop
- durable scheduling
- artifact promotion between PC1 and PC2
- deploy-safe inference session runner
- stronger monitoring and failure recovery
