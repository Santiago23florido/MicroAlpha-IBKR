# MicroAlpha-IBKR Phase 3 Data Pipeline

## Purpose

The repository is organized around a two-machine workflow:

- `PC1`: research, data validation, feature engineering, model training, backtesting
- `PC2`: IBKR connectivity, market data collection, operational monitoring

Phase 1 established the repository structure and configuration model.
Phase 2 added the first operational collector on `PC2`.
Phase 3 adds the `PC1` data pipeline that turns collected raw market data into validated, cleaned, feature-rich parquet datasets for later model work.

This phase does **not** implement autonomous trading, final models, order execution, RL, LLM/news, or advanced optimization.

## Phase 3 Scope

Phase 3 adds:

- structured loading of raw market parquet partitions
- data-quality validation and issue reporting
- deterministic cleaning and normalization
- ORB, microstructure, intraday, and cost features
- parquet persistence for processed features
- a dataset builder for future training workflows
- a CLI entrypoint for end-to-end feature generation

## PC2 -> PC1 Flow

Current intended workflow:

1. `PC2` runs the collector and stores raw parquet under `data/raw/market/`
2. raw partitions are synced from `PC2` to `PC1`
3. `PC1` runs `python app.py build-features`
4. the pipeline validates, cleans, and engineers features
5. processed parquet is written under `data/features/`
6. later phases will build labels, training jobs, and inference on top of that output

## Project Structure

```text
MicroAlpha-IBKR/
├── app.py
├── config/
│   ├── settings.yaml
│   ├── risk.yaml
│   ├── symbols.yaml
│   └── deployment.yaml
├── data/
│   ├── loader.py
│   ├── cleaning.py
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
├── features/
│   ├── orb_features.py
│   ├── microstructure_features.py
│   ├── preprocessing.py
│   └── feature_pipeline.py
├── labels/
│   ├── generator.py
│   └── dataset_builder.py
├── monitoring/
│   ├── logging.py
│   ├── healthcheck.py
│   └── data_quality.py
├── scripts/
│   ├── run_collector.py
│   ├── build_features.py
│   ├── train_model.py
│   ├── run_session.py
│   ├── healthcheck.py
│   └── sync_data.py
└── tests/
```

Existing modules from earlier phases remain in place:

- `broker/`
- `engine/`
- `models/`
- `risk/`
- `strategy/`
- `storage/`
- `ui/`

## Configuration

Configuration is merged from:

- `.env`
- `config/settings.yaml`
- `config/risk.yaml`
- `config/symbols.yaml`
- `config/deployment.yaml`

Environment modes:

- `development` for `PC1`
- `deploy` for `PC2`

### Relevant Phase 3 Settings

From `config/settings.yaml` and `.env`:

- `DATA_ROOT`
- `MARKET_RAW_DIR`
- `SUPPORTED_SYMBOLS`
- `LOG_LEVEL`
- `TIMEZONE`
- `FEATURE_GAP_THRESHOLD_SECONDS`
- `FEATURE_MAX_ABS_SPREAD_BPS`
- `FEATURE_FORWARD_FILL_LIMIT`
- `FEATURE_DROP_OUTSIDE_REGULAR_HOURS`
- `FEATURE_ROLLING_SHORT_WINDOW`
- `FEATURE_ROLLING_MEDIUM_WINDOW`
- `FEATURE_ROLLING_LONG_WINDOW`
- `FEATURE_VWAP_WINDOW`
- `FEATURE_VOLUME_WINDOW`
- `FEATURE_LABEL_HORIZON_ROWS`
- `FEATURE_TRAIN_SPLIT_RATIO`

Inspect the effective config with:

```bash
python app.py show-config
python app.py --environment development show-config
python app.py --environment deploy show-config
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Raw Data Layout

The collector on `PC2` writes raw market data under:

```text
data/raw/market/YYYY-MM-DD/SYMBOL/collector_*.parquet
```

The phase 3 loader also accepts the flatter variant:

```text
data/raw/market/YYYY-MM-DD/SYMBOL.parquet
```

Expected raw columns are normalized around:

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

## Feature Output Layout

Processed features are written to:

```text
data/features/YYYY-MM-DD/SYMBOL.parquet
```

Feature-build reports are written to:

```text
data/reports/feature_build_report_<timestamp>.json
```

## Features Implemented

### ORB Features

- `orb_high`
- `orb_low`
- `orb_range_mid`
- `orb_range_width`
- `orb_range_width_bps`
- `orb_relative_price_position`
- `breakout_distance`
- `breakout_distance_bps`
- `orb_range_complete`
- `minutes_since_open`

### Microstructure Features

- `mid_price`
- `micro_price`
- `spread`
- `spread_bps`
- `bid_ask_imbalance`
- `rolling_spread_mean_bps`
- `rolling_spread_std_bps`
- `rolling_imbalance_mean`
- `rolling_imbalance_std`

### Intraday Features

- `return_1_bps`
- `return_short_bps`
- `return_medium_bps`
- `rolling_volatility_short_bps`
- `rolling_volatility_medium_bps`
- `rolling_volatility_long_bps`
- `vwap_approx`
- `distance_to_vwap_bps`
- `relative_volume`
- `time_of_day_sin`
- `time_of_day_cos`

### Cost Features

- `spread_proxy_bps`
- `slippage_proxy_bps`
- `estimated_cost_bps`

## Data Quality Validation

The validation layer checks for:

- missing timestamps
- duplicated rows
- large timestamp gaps
- critical nulls
- `bid > ask`
- negative spreads
- absurd spreads in bps
- rows outside regular market hours

The feature pipeline logs detected issues before and after cleaning so you can see whether cleaning fixed them or whether they remain in the source data.

## Cleaning Rules

Cleaning currently does the following:

- enforce timestamp parsing and numeric types
- sort by `symbol` and `timestamp`
- drop duplicate `(symbol, timestamp)` rows
- forward fill quote fields with a bounded limit
- recompute `mid_price` and `spread` from cleaned quotes
- discard invalid quotes and absurd spreads
- optionally drop rows outside regular hours
- attach session metadata used by feature generation

The cleaning layer is intentionally conservative. It avoids inventing values beyond bounded forward fill.

## Main Commands

### Build Features on PC1

Default run on configured symbols:

```bash
python app.py build-features
```

Date-bounded run:

```bash
python app.py build-features --start-date 2026-04-01 --end-date 2026-04-05
```

Symbol override:

```bash
python app.py build-features --symbols SPY QQQ
```

Custom roots:

```bash
python app.py build-features \
  --input-root /path/to/raw_market \
  --output-root /path/to/feature_output
```

Script entrypoint:

```bash
python scripts/build_features.py --symbols SPY QQQ
```

### Collector on PC2

One bounded cycle:

```bash
python app.py --environment deploy collect --once
```

Longer deploy-side run:

```bash
python app.py --environment deploy collect --max-cycles 120
```

### Healthcheck

Config and paths only:

```bash
python app.py healthcheck --skip-broker
```

Collector-oriented broker check:

```bash
python app.py --environment deploy healthcheck
```

Script form:

```bash
python scripts/healthcheck.py --environment deploy --skip-broker
```

## Example Workflow

### PC2

Collect raw market data:

```bash
python app.py --environment deploy collect --max-cycles 60
```

### PC1

Build research features from the synced raw data:

```bash
python app.py --environment development build-features --symbols SPY QQQ
```

## Dataset Builder

`labels/dataset_builder.py` prepares the next step for training workflows:

- selects numeric feature columns
- creates a placeholder future-return target
- keeps temporal ordering intact
- produces a temporal train/test split

This is intentionally a base layer, not the final production labeling design.

## Logging

Shared logging comes from `monitoring/logging.py`.

Phase 3 logs at least:

- pipeline start and finish
- raw rows loaded
- symbols and day range processed
- quality issues detected
- cleaned row counts
- feature row counts
- written parquet files
- report path and runtime duration

Default log location:

```text
data/logs/microalpha.log
```

## Current Limitations

Phase 3 is intentionally limited:

- no final labels yet
- no advanced outlier correction
- no feature-store versioning beyond parquet partitions
- no online feature computation on `PC2`
- no inference or execution
- no cross-day session stitching logic for multi-session products
- no advanced sync orchestration between machines

## What Phase 4 Should Add

Phase 4 can now build on this data foundation:

- richer label generation
- training datasets tied to explicit prediction horizons
- stronger data-quality reports and dashboards
- feature-store validation and versioning rules
- online feature parity between `PC2` and `PC1`
- inference-preparation pipeline

## Tests

Run the full suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests
```

Phase 3 added tests for:

- raw parquet loading across multiple days and layouts
- data-quality issue detection
- cleaning + feature build end to end
- dataset builder temporal splits
