# MicroAlpha-IBKR Phase 4 LAN Transfer and Data Pipeline

## Purpose

This repository is organized around a two-machine setup on the same local network:

- `PC2`: IBKR collector and operational data source
- `PC1`: research, validation, feature engineering, model training, and evaluation

Phase 1 organized the repository and CLI.
Phase 2 added the collector on `PC2`.
This unified Phase 4 replaces the old cloud-sync idea with a local-network workflow:

1. `PC2` writes raw market data locally
2. `PC1` pulls new files from a shared network path on `PC2`
3. `PC1` validates imported data
4. `PC1` cleans and transforms imported raw data into feature parquet files

This phase does **not** implement autonomous trading, final execution logic, or cloud sync.

## Architecture

### PC2

- runs IBKR Gateway or TWS
- runs the market data collector
- persists operational raw parquet locally under `data/raw/market/`
- optionally exposes `data/meta/` and `data/logs/` through the same network share

### PC1

- mounts or accesses the shared folder from `PC2`
- imports new files into `imports/from_pc2/`
- validates imported parquet files
- builds features into `data/features/`

The intended ownership is:

- `PC2` = origin of truth for collection
- `PC1` = local working copy for research

## Project Structure

```text
MicroAlpha-IBKR/
├── app.py
├── config/
│   ├── settings.yaml
│   ├── risk.yaml
│   ├── symbols.yaml
│   └── deployment.yaml
├── deployment/
│   └── lan_sync.py
├── data/
│   ├── loader.py
│   ├── cleaning.py
│   ├── raw/
│   │   └── market/
│   ├── features/
│   ├── processed/
│   ├── logs/
│   └── reports/
├── imports/
│   └── from_pc2/
│       ├── raw/
│       │   └── market/
│       ├── meta/
│       ├── logs/
│       └── transfer_log.jsonl
├── ingestion/
│   └── collector.py
├── features/
│   └── feature_pipeline.py
├── monitoring/
│   ├── healthcheck.py
│   ├── logging.py
│   └── data_quality.py
├── scripts/
│   ├── run_collector.py
│   ├── pull_from_pc2.py
│   ├── validate_imports.py
│   ├── build_features.py
│   ├── dev_sync_and_build.py
│   └── healthcheck.py
└── tests/
```

## Data Layout

### Source layout on PC2

The collector is expected to write under a local project root on `PC2`:

```text
data/raw/market/YYYY-MM-DD/SYMBOL/*.parquet
data/meta/
data/logs/
```

### Import layout on PC1

Pulled files are copied into the local research workspace:

```text
imports/from_pc2/raw/market/YYYY-MM-DD/SYMBOL/*.parquet
imports/from_pc2/meta/
imports/from_pc2/logs/
imports/from_pc2/transfer_log.jsonl
```

### Feature layout on PC1

```text
data/features/YYYY-MM-DD/SYMBOL.parquet
```

## Network Share Configuration

This implementation assumes `PC1` can access a shared filesystem path from `PC2`.
The application does **not** mount SMB shares itself; it works with an already reachable path.

Examples:

- Windows UNC path:

```text
\\PC2\microalpha
```

- Mounted SMB share on Linux or WSL:

```text
/mnt/pc2/microalpha
```

- Mounted network drive letter from Windows exposed into WSL:

```text
/mnt/z/microalpha
```

Set that location through `PC2_NETWORK_ROOT`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Configuration

Inspect the merged configuration:

```bash
python app.py show-config
python app.py --environment development show-config
python app.py --environment deploy show-config
```

The most important LAN settings are:

- `PC2_NETWORK_ROOT`
- `IMPORT_ROOT`
- `IMPORT_MARKET_DIR`
- `LAN_INCLUDE_RAW`
- `LAN_INCLUDE_META`
- `LAN_INCLUDE_LOGS`
- `LAN_DRY_RUN`
- `LAN_OVERWRITE_POLICY`
- `LAN_VALIDATE_PARQUET`
- `LAN_ALLOWED_SYMBOLS`

## Main Commands

### PC2 collector

Run on `PC2`:

```bash
python app.py --environment deploy collect --once
python app.py --environment deploy collect --max-cycles 120
```

### Pull files from PC2 to PC1

Run on `PC1`:

```bash
python app.py --environment development pull-from-pc2
python app.py --environment development pull-from-pc2 --symbols SPY QQQ
python app.py --environment development pull-from-pc2 --start-date 2026-04-14 --end-date 2026-04-16
python app.py --environment development pull-from-pc2 --dry-run
```

Wrapper script:

```bash
python scripts/pull_from_pc2.py --environment development
```

### Validate imports on PC1

```bash
python app.py --environment development validate-imports
python app.py --environment development validate-imports --symbols SPY
```

Wrapper script:

```bash
python scripts/validate_imports.py --environment development
```

### Build features on PC1

By default this reads from `imports/from_pc2/raw/market/`.

```bash
python app.py --environment development build-features
python app.py --environment development build-features --symbols SPY QQQ
python app.py --environment development build-features --start-date 2026-04-14 --end-date 2026-04-16
```

Wrapper script:

```bash
python scripts/build_features.py --environment development
```

### One-command development flow

This is the main convenience command for `PC1`:

```bash
python app.py --environment development dev-sync-and-build
python app.py --environment development dev-sync-and-build --symbols SPY
python app.py --environment development dev-sync-and-build --dry-run
```

Wrapper script:

```bash
python scripts/dev_sync_and_build.py --environment development
```

### Healthcheck

```bash
python app.py healthcheck --skip-broker
python app.py --environment deploy healthcheck
```

## Transfer Tracking

Every pull from `PC2` leaves local traceability on `PC1`:

- incremental event log:

```text
imports/from_pc2/transfer_log.jsonl
```

- per-run JSON reports:

```text
data/reports/lan_sync/pull_from_pc2_*.json
```

Each record includes:

- source path
- destination path
- category
- file size
- modification timestamp
- transfer status
- validation result

This keeps a simple but useful history of what was detected, copied, skipped, or failed.

## Import Validation

`validate-imports` checks:

- file readability
- required columns
- obvious duplicate rows
- missing timestamps
- critical null rows
- `bid > ask`
- negative or absurd spreads
- large gaps
- rows outside regular hours

Validation reports are written into `data/reports/`.

## Feature Pipeline

The feature pipeline generates four groups of features.

### ORB features

- `orb_high`
- `orb_low`
- `orb_range_width`
- `orb_range_width_bps`
- `orb_relative_price_position`
- `breakout_distance`
- `breakout_distance_bps`
- `orb_range_complete`
- `minutes_since_open`

### Microstructure features

- `spread`
- `spread_bps`
- `mid_price`
- `micro_price`
- `bid_ask_imbalance`
- `rolling_spread_mean_bps`
- `rolling_spread_std_bps`
- `rolling_imbalance_mean`
- `rolling_imbalance_std`

### Intraday features

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

### Cost proxy features

- `estimated_cost_bps`
- `spread_proxy_bps`
- `slippage_proxy_bps`

## Windows / SMB Notes

Typical setup for a Windows-based `PC2`:

1. share the project folder or a data subfolder from Windows
2. grant read access to the user that will connect from `PC1`
3. from `PC1`, use either:
   - the UNC path directly if the runtime supports it
   - a mounted share path
   - a mapped network drive available inside WSL as `/mnt/<drive>/...`

Example:

```dotenv
PC2_NETWORK_ROOT=/mnt/z/microalpha
```

where `Z:` is a mapped drive pointing to:

```text
\\PC2\microalpha
```

## Limitations

- the application assumes the network share is already reachable
- it does not mount SMB shares or manage credentials
- transfer change detection is based on path, size, and modified time, not content hashing
- import validation is basic but practical; it is not a full market data QA framework
- feature generation is offline research preparation, not online inference

## Next Phase

The next logical phase is:

- labels and dataset generation from `data/features/`
- temporal training / validation splits
- baseline model training and evaluation
- later, alignment between offline features and online inference
