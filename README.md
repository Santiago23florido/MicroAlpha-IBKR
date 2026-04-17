# MicroAlpha-IBKR Phase 6: Inference, Decision, and Risk on Top of the LAN Modeling Pipeline

## Purpose

This repository now supports the Phase 6 operational layer on top of the Phase 5 modeling workflow while keeping the dual-machine architecture intact:

- `PC2`: collector and operational raw data source
- `PC1`: LAN import, validation, feature engineering, labeling, dataset building, model comparison, active-model selection, inference, decision, and operational risk checks

Phase 6 still does **not** send real orders. It connects:

- feature stores
- one explicitly selected active model from Phase 5
- a normalized inference layer
- a decision engine
- a risk engine
- structured decision logs
- offline and session runners without order routing

The active model is visible in `config/active_model.yaml`, not hidden in code.

## Architecture

### PC2

- runs the collector
- writes raw parquet under `data/raw/market/`
- exposes the project or data folder through a LAN share

### PC1

- pulls data from `PC2` into `imports/from_pc2/`
- validates imports
- builds configurable feature stores
- builds labels from those feature stores
- builds modeling datasets with temporal splits
- trains multiple model variants
- evaluates them and writes a leaderboard plus artifacts

## Project Structure

```text
MicroAlpha-IBKR/
├── app.py
├── config/
│   ├── settings.yaml
│   ├── feature_sets.yaml
│   ├── modeling.yaml
│   ├── active_model.yaml
│   ├── phase6.yaml
│   ├── risk.yaml
│   ├── symbols.yaml
│   └── deployment.yaml
├── deployment/
│   └── lan_sync.py
├── engine/
│   └── phase6.py
├── data/
│   ├── loader.py
│   ├── cleaning.py
│   ├── feature_loader.py
│   ├── features/
│   ├── processed/
│   │   └── labels/
│   ├── models/
│   └── reports/
│       ├── phase5/
│       ├── phase6/
│       └── decisions/
├── features/
│   ├── definitions.py
│   ├── registry.py
│   ├── validation.py
│   ├── feature_pipeline.py
│   └── indicators/
├── labels/
│   ├── labeling.py
│   └── dataset_builder.py
├── models/
│   ├── config.py
│   ├── factory.py
│   ├── evaluation.py
│   ├── experiments.py
│   ├── registry.py
│   ├── inference.py
│   ├── train_baseline.py
│   └── train_deep.py
├── risk/
│   └── risk_engine.py
├── strategy/
│   ├── decision_engine.py
│   └── explainability.py
├── storage/
│   └── decision_logs.py
├── imports/
│   └── from_pc2/
├── scripts/
│   ├── pull_from_pc2.py
│   ├── validate_imports.py
│   ├── build_features.py
│   ├── build_labels.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── compare_model_variants.py
│   ├── run_phase5_experiments.py
│   ├── show_active_model.py
│   ├── set_active_model.py
│   ├── run_decisions_offline.py
│   └── risk_check.py
└── tests/
```

## LAN Data Flow

1. `PC2` writes raw parquet locally.
2. `PC1` reaches the LAN share configured by `PC2_NETWORK_ROOT`.
3. `PC1` runs `pull-from-pc2`.
4. Imported parquet lands in `imports/from_pc2/raw/market/`.
5. `PC1` runs `build-features --feature-set <name>`.
6. Features are written to `data/features/<feature_set>/YYYY-MM-DD/SYMBOL.parquet`.
7. `PC1` runs `build-labels`.
8. Labels are written to `data/processed/labels/<feature_set>/<target_mode>/YYYY-MM-DD/SYMBOL.parquet`.
9. `PC1` trains and compares model variants.
10. `PC1` selects one active model for operational inference.
11. `PC1` runs offline or session decisions without sending orders.

`PC2` remains the operational source of truth for collection.  
`PC1` remains the research and experiment node.

## Network Share Configuration

The repo does not mount SMB or UNC shares for you. `PC1` must already be able to access the filesystem exported by `PC2`.

Examples:

- Windows UNC:

```text
\\PC2\microalpha
```

- Linux or WSL mounted share:

```text
/mnt/pc2/microalpha
```

- Mounted Windows drive visible in WSL:

```text
/mnt/z/microalpha
```

Set that root through `PC2_NETWORK_ROOT`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Phase 5 and Phase 6 Architecture

Phase 5 remains the modeling layer:

1. feature / indicator system
2. labeling / target system
3. dataset building
4. model factory
5. variant and hyperparameter search
6. evaluation and leaderboard
7. model registry and artifacts
8. simple execution commands

Phase 6 adds:

1. active model selection
2. uniform model loading and inference
3. normalized prediction output
4. decision engine
5. risk engine
6. position sizing
7. explainability
8. offline decision runner
9. session runner without orders
10. structured decision logging

### Feature Registry

The feature registry lives in `features/registry.py`.

Each indicator declares:

- canonical name
- family
- required dependency groups
- configurable parameters
- output columns
- output type
- calculator implementation

The registry is dependency-aware. If a dataset does not have usable columns for an indicator, the indicator is skipped with an explicit reason instead of failing ambiguously.

### Indicator Families

#### Trend

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

#### Momentum

- `rsi`
- `roc`
- `momentum_simple`
- `stochastic_k`
- `stochastic_d`
- `williams_r`
- `cci`

#### Volatility

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

#### Volume / Flow

- `rolling_volume_mean`
- `relative_volume`
- `vwap`
- `distance_to_vwap`
- `vwap_slope`
- `obv`
- `volume_spike_flag`
- `accumulation_distribution`
- `mfi`

#### Microstructure

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

#### Intraday Structure

- `minute_of_day`
- `seconds_since_open`
- `seconds_to_close`
- `opening_session_flag`
- `midday_flag`
- `closing_session_flag`
- `day_of_week`
- `intraday_volume_percentile`
- `intraday_spread_percentile`

### Feature Sets

Defined in `config/feature_sets.yaml`:

- `core_price_only`
- `core_intraday`
- `technical_basic`
- `technical_plus_volume`
- `microstructure_core`
- `hybrid_intraday`
- `full_experimental`

Each feature set declares:

- families
- explicit indicators
- optional parameter overrides
- minimum columns

### Target Modes

Defined in `config/modeling.yaml`:

- `classification_binary`
  - binary continuation target from future net return vs threshold
- `regression_point`
  - future return in bps
- `ordinal_classification`
  - ordered return bins
- `distribution_bins`
  - discrete return buckets as a distribution-oriented classification target
- `quantile_regression`
  - continuous future return target prepared for multi-quantile regressors

The pipeline separates `X` and `y` cleanly:

- features only use present and past information
- labels use future information only in target construction
- the dataset builder excludes `target_*`, `future_*`, and other leakage-prone columns from feature selection

### Supported Models

Built through `models/factory.py`.

#### Classification / Ordinal / Distribution Bins

- `logistic_regression`
- `random_forest_classifier`
- `hist_gradient_boosting_classifier`
- `xgboost_classifier` if installed
- `lightgbm_classifier` if installed

#### Point Regression

- `ridge_regression`
- `random_forest_regressor`
- `hist_gradient_boosting_regressor`
- `xgboost_regressor` if installed
- `lightgbm_regressor` if installed

#### Distribution / Quantile-Oriented

- `quantile_gradient_boosting`
  - trains multiple quantile regressors, for example `q10 / q50 / q90`
- `distribution_bins`
  - discrete return buckets through multiclass classification

### Variant Search

The search configuration lives in `config/modeling.yaml`.

It controls:

- temporal split ratios
- minimum sample thresholds
- target definitions
- model parameter grids
- experiment profiles

The search is intentionally simple:

- practical grids
- explicit combinations
- reproducible runs
- no expensive tuning machinery

### Temporal Evaluation

The main split is temporal:

- `train`
- `validation`
- `test`

No random split is used as the primary validation path.

### Metrics

#### Technical

Classification and ordinal:

- accuracy
- precision macro
- recall macro
- F1 macro
- weighted F1
- ROC AUC when applicable
- confusion matrix

Regression:

- MAE
- RMSE
- directional accuracy

Quantile / distribution-oriented:

- pinball loss
- interval coverage
- mean interval width

#### Preliminary Economic

- top-decile mean future return
- top-decile mean net return
- bottom-decile mean future return
- score spread between top and bottom deciles
- top-signal hit rate
- score buckets with mean future return

## Core Commands

### LAN and Features

```bash
python app.py pull-from-pc2
python app.py validate-imports
python app.py list-feature-sets
python app.py inspect-feature-dependencies --feature-set hybrid_intraday
python app.py build-features --feature-set hybrid_intraday
python app.py validate-features --feature-set hybrid_intraday
```

### Labels and Single-Run Modeling

```bash
python app.py build-labels --feature-set hybrid_intraday --target-mode classification_binary
python app.py train-baseline --feature-set hybrid_intraday --target-mode classification_binary --model logistic_regression
python app.py evaluate-baseline
```

### Comparison and Main Phase 5 Runner

```bash
python app.py compare-model-variants --profile default
python app.py run-phase5-experiments --profile default
```

### Phase 6 Operational Commands

```bash
python app.py show-active-model
python app.py set-active-model --run-id <phase5_run_id>
python app.py risk-check
python app.py run-decisions-offline --symbols SPY --limit 200
python app.py run-session --symbols SPY
```

## Active Model Selection

The default operational model is stored in:

```text
config/active_model.yaml
```

That file declares:

- `run_id`
- `artifact_dir`
- `model_name`
- `model_type`
- `feature_set_name`
- `target_mode`
- `selection_reason`

Change it with:

```bash
python app.py set-active-model --run-id <phase5_run_id>
```

or, if you want the best run for a given model family:

```bash
python app.py set-active-model --model-name logistic_regression
```

Review the current selection with:

```bash
python app.py show-active-model
```

## Phase 6 Inference and Decisions

`models/inference.py` loads the active Phase 5 artifact and normalizes output into one structure for the operational layer. Depending on the target mode, it can emit:

- `score`
- `probability`
- `predicted_return_bps`
- `predicted_quantiles`
- `class_label`
- model metadata

`strategy/decision_engine.py` then converts that prediction into:

- `LONG`
- `SHORT`
- `NO_TRADE`

plus:

- confidence
- expected return proxy
- expected cost
- net edge
- size suggestion
- reasons

`risk/risk_engine.py` applies:

- max trades per session
- daily loss limit
- symbol loss limit
- cooldown after loss
- spread and estimated-cost limits
- kill switch on invalid/anomalous model output

## Offline and Session Runners

Offline review:

```bash
python app.py run-decisions-offline --symbols SPY --limit 200
```

This command:

1. loads the active feature set
2. loads the active model
3. runs inference row by row
4. applies decision rules
5. applies risk rules
6. writes JSONL plus CSV/Parquet summaries under `data/reports/phase6/`

Session review:

```bash
python app.py run-session --symbols SPY
```

This command reads the latest available feature rows and runs the same decision and risk logic, but:

- does **not** send orders
- does **not** paper trade yet
- only logs what the current operational layer would decide

### Examples

Run one distribution-oriented comparison:

```bash
python app.py compare-model-variants \
  --feature-sets hybrid_intraday \
  --target-modes quantile_regression \
  --models quantile_gradient_boosting \
  --symbols SPY
```

Run the main experiment command with explicit scope:

```bash
python app.py run-phase5-experiments \
  --feature-sets hybrid_intraday technical_plus_volume \
  --target-modes classification_binary regression_point quantile_regression \
  --models logistic_regression ridge_regression quantile_gradient_boosting \
  --symbols SPY QQQ
```

Reuse an existing feature store and skip feature regeneration:

```bash
python app.py run-phase5-experiments \
  --feature-sets hybrid_intraday \
  --target-modes classification_binary \
  --models logistic_regression \
  --skip-feature-build
```

## Leaderboard

Every comparison run writes a leaderboard under `data/reports/phase5/`.

Outputs include:

- `leaderboard_*.json`
- `leaderboard_*.csv`
- `leaderboard_*.parquet`

Each row contains:

- `run_id`
- timestamp
- model name
- target mode
- feature set
- hyperparameters
- split config
- symbols
- train / validation / test ranges
- validation metrics
- test metrics
- artifact path
- ranking score

Interpretation:

- use the leaderboard to compare runs
- do **not** treat rank 1 as “the final production model”
- check both technical metrics and economic separation
- compare the same model across feature sets
- compare different model families on the same target mode

## Artifacts and Registry

Each run writes a dedicated artifact directory under `data/models/`:

```text
data/models/run_<timestamp>_<model>_<feature_set>_<target>_<id>/
  model.joblib
  preprocessing.joblib
  feature_columns.json
  metrics.json
  config_snapshot.json
  training_metadata.json
  target_config.json
  leaderboard_row.json
  evaluation.json
```

The registry keeps:

- legacy `baseline` and `deep` entries for old flows
- `phase5_runs` for the new flexible modeling system

## Validations and Safety Checks

The pipeline fails clearly when:

- the selected feature set cannot be built
- required feature store files are missing
- labels cannot be built because no valid price proxy exists
- feature columns become empty, constant, or excessively sparse
- the target is missing or invalid
- the temporal split is too small or empty
- the selected model is incompatible with the target mode
- a model artifact would be saved without required metadata

## Limitations

- Phase 6 does not send real orders yet.
- It also does not implement final paper trading.
- Binary classification targets can only express `LONG` or `NO_TRADE` cleanly; they are not a true short target.
- The session runner currently operates on the latest available feature rows, not on a live streaming feature service.
- The risk engine is prepared for realized PnL tracking, but without broker integration it only uses offline realized outcomes when labels are available.
- No portfolio optimization or multi-position allocation logic is included yet.
- Optional `xgboost` and `lightgbm` support still depends on those packages being installed.

## Phase 7

The next phase should build on this by adding:

- paper trading with broker integration gates
- stronger session state and realized PnL plumbing
- decision-to-order translation
- execution audit and fill reconciliation
- monitoring and drift checks
- promotion rules for switching the active model safely
