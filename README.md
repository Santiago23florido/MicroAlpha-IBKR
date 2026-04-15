# MicroAlpha-IBKR Phase 5A: Flexible Feature Architecture on Top of the LAN Pipeline

## Purpose

The repository keeps the same dual-machine LAN architecture:

- `PC2`: IBKR collector and operational raw data source
- `PC1`: research workstation for imports, validation, feature generation, and later model comparison

This subsection of Phase 5 does **not** implement final modeling, execution, or decision logic.
It only prepares the feature layer so the project can:

- define indicator families centrally
- enable or disable feature sets by configuration
- detect dependency compatibility against the current dataset
- skip incompatible indicators with explicit reasons
- rebuild comparable feature variants when the dataset changes

## Architecture

### PC2

- runs the collector
- writes raw parquet under `data/raw/market/`
- exposes that data through a LAN share

### PC1

- pulls data from the LAN share into `imports/from_pc2/`
- validates imports
- cleans raw data
- applies a configurable feature set through the feature registry
- writes feature parquet files under `data/features/`

## Project Structure

```text
MicroAlpha-IBKR/
├── app.py
├── config/
│   ├── settings.yaml
│   ├── feature_sets.yaml
│   ├── risk.yaml
│   ├── symbols.yaml
│   └── deployment.yaml
├── deployment/
│   └── lan_sync.py
├── data/
│   ├── loader.py
│   ├── cleaning.py
│   ├── features/
│   └── reports/
├── features/
│   ├── definitions.py
│   ├── registry.py
│   ├── validation.py
│   ├── feature_pipeline.py
│   └── indicators/
│       ├── trend.py
│       ├── momentum.py
│       ├── volatility.py
│       ├── volume_flow.py
│       ├── microstructure.py
│       └── intraday.py
├── imports/
│   └── from_pc2/
├── monitoring/
│   ├── healthcheck.py
│   ├── logging.py
│   └── data_quality.py
├── scripts/
│   ├── pull_from_pc2.py
│   ├── validate_imports.py
│   ├── build_features.py
│   ├── dev_sync_and_build.py
│   ├── list_feature_sets.py
│   ├── inspect_feature_dependencies.py
│   └── validate_features.py
└── tests/
```

## LAN Data Flow

The operational flow is still LAN-first.

1. `PC2` writes raw market parquet locally.
2. `PC1` mounts or reaches the shared folder from `PC2`.
3. `PC1` runs `pull-from-pc2`.
4. Imported parquet lands in `imports/from_pc2/raw/market/`.
5. `PC1` runs `build-features --feature-set <name>`.
6. Features are written to `data/features/YYYY-MM-DD/SYMBOL.parquet`.

`PC2` remains the operational source of truth for collection.
`PC1` keeps the working copy for research.

## Network Share Configuration

The code assumes `PC1` can already reach the filesystem exported by `PC2`.
It does **not** mount SMB shares itself.

Examples:

- Windows UNC path:

```text
\\PC2\microalpha
```

- Mounted share in Linux or WSL:

```text
/mnt/pc2/microalpha
```

- Mounted Windows drive visible from WSL:

```text
/mnt/z/microalpha
```

Configure that path through `PC2_NETWORK_ROOT`.

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

Most important LAN settings:

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

Most important feature-architecture settings:

- `FEATURE_SET`
- `FEATURE_ROLLING_SHORT_WINDOW`
- `FEATURE_ROLLING_MEDIUM_WINDOW`
- `FEATURE_ROLLING_LONG_WINDOW`
- `FEATURE_VWAP_WINDOW`
- `FEATURE_VOLUME_WINDOW`
- `FEATURE_VALIDATION_MAX_NAN_RATIO`

## Feature Registry

The feature registry lives in `features/registry.py`.

Each indicator declares:

- canonical name
- family
- required dependency groups
- default parameters
- output columns
- output type
- calculator implementation

The registry does not assume the dataset always exposes the same columns.
Instead it checks whether each indicator has at least one usable column for every dependency group.

Example:

- `rsi` needs a usable price proxy
- `vwap` needs price proxy plus volume
- `microprice_proxy` needs `bid`, `ask`, `bid_size`, `ask_size`

If those dependencies are not present with usable data, the indicator is omitted and the reason is recorded.

## Indicator Families

### Trend

- `sma`
- `ema`
- `moving_average_distance`
- `moving_average_slope`
- `ma_crossover_short_long`
- `macd_line`
- `macd_signal`
- `macd_histogram`
- `adx`
- `plus_di`
- `minus_di`

### Momentum

- `rsi`
- `roc`
- `momentum_simple`
- `stochastic_k`
- `stochastic_d`
- `williams_r`
- `cci`

### Volatility

- `true_range`
- `atr`
- `rolling_volatility`
- `rolling_std_returns`
- `bollinger_mid`
- `bollinger_upper`
- `bollinger_lower`
- `bollinger_bandwidth`
- `zscore_price`
- `orb_width`

### Volume / Flow

- `rolling_volume_mean`
- `relative_volume`
- `vwap`
- `distance_to_vwap`
- `vwap_slope`
- `obv`
- `volume_spike_flag`
- `accumulation_distribution`
- `mfi`

### Microstructure

- `spread`
- `spread_bps`
- `mid_price`
- `weighted_mid_price`
- `imbalance`
- `rolling_imbalance`
- `delta_imbalance`
- `total_depth`
- `depth_ratio`
- `microprice_proxy`

### Intraday Structure

- `minute_of_day`
- `seconds_since_open`
- `seconds_to_close`
- `opening_session_flag`
- `midday_flag`
- `closing_session_flag`
- `day_of_week`
- `intraday_volume_percentile`
- `intraday_spread_percentile`

The pipeline also keeps a small set of compatibility features such as `return_1_bps`, `return_short_bps`, `return_medium_bps`, `estimated_cost_bps`, `spread_proxy_bps`, and `slippage_proxy_bps`.

## Feature Sets

Feature sets are declared centrally in `config/feature_sets.yaml`.

Available sets:

- `core_price_only`
- `core_intraday`
- `technical_basic`
- `technical_plus_volume`
- `microstructure_core`
- `hybrid_intraday`
- `full_experimental`

Each set declares:

- description
- families
- explicit indicators
- parameter overrides when needed
- minimum expected columns

This lets you compare later:

- same model with different feature sets
- different models on the same feature set

without rewriting the build pipeline.

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

### Validate imports

```bash
python app.py --environment development validate-imports
python app.py --environment development validate-imports --symbols SPY
```

### List feature sets

```bash
python app.py list-feature-sets
python scripts/list_feature_sets.py --environment development
```

### Inspect feature dependencies

```bash
python app.py inspect-feature-dependencies --feature-set hybrid_intraday
python app.py inspect-feature-dependencies --feature-set microstructure_core --symbols SPY
python scripts/inspect_feature_dependencies.py --environment development --feature-set technical_plus_volume
```

### Build features with a selected set

```bash
python app.py build-features --feature-set hybrid_intraday
python app.py build-features --feature-set technical_basic --symbols SPY QQQ
python app.py build-features --feature-set microstructure_core --start-date 2026-04-14 --end-date 2026-04-16
python scripts/build_features.py --environment development --feature-set hybrid_intraday
```

### Validate generated features

```bash
python app.py validate-features
python app.py validate-features --symbols SPY
python scripts/validate_features.py --environment development
```

### One-command development flow

```bash
python app.py --environment development dev-sync-and-build --feature-set hybrid_intraday
python app.py --environment development dev-sync-and-build --symbols SPY --feature-set technical_plus_volume
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

## Feature Metadata and Validation

Each `build-features` run writes:

- `data/reports/feature_manifest_*.json`
- `data/reports/feature_build_report_*.json`

The manifest/report pair stores:

- selected feature set
- families activated
- indicators calculated
- indicators omitted
- omission reasons
- parameters used
- input columns detected
- feature columns emitted
- symbol/date range
- raw quality summary
- cleaned quality summary
- feature quality summary

`validate-features` checks for:

- empty feature columns
- constant columns
- excessive NaNs
- infinities
- duplicate columns

## Limitations

- The loader still standardizes around the current raw market schema, even though it now preserves extra columns if present.
- Dependency detection is column-and-non-null based; it does not yet score feature usefulness.
- No model training comparison is implemented in this subsection.
- No automatic feature-selection policy exists yet.
- Feature validation reports data health, but it does not yet gate model training.

## Next Step After This Subsection

The next subsection of Phase 5 should build on this layer to add:

- labels
- dataset assembly per feature set
- baseline training/evaluation with temporal splits
- artifact metadata tying models to the exact feature set and manifest used
