# MicroAlpha-IBKR Phase 12 to 14: Deployment, Shadow Mode, Model Lifecycle, Governance, and Safe Paper Operations on Top of the LAN Modeling Pipeline

## Purpose

This repository now supports the combined Phase 12, Phase 13, and Phase 14 layer on top of the Phase 5, Phase 6, Phase 7, Phase 8, Phase 9, and Phase 10/11 workflow while keeping the dual-machine architecture intact:

- `PC2`: collector and operational raw data source
- `PC1`: LAN import, validation, feature engineering, labeling, dataset building, model comparison, active-model selection, inference, decision, risk, paper execution, order journaling, and execution status

The system keeps the same normalized operational chain and now adds deployment/runtime profiles, stable PC2 bootstrap, shadow mode, release governance, promotion/rollback, runtime control, and a Polygon Basic bootstrap training-data path without enabling any live-trading path:

- feature stores
- one explicitly selected active model from Phase 5
- a normalized inference layer
- a decision engine
- a risk engine
- an order manager
- a paper/mock execution backend
- an IBKR Paper execution backend compatible with the same interface
- a configurable fill simulator
- paper positions and PnL tracking
- structured journals for orders, fills, positions, and PnL
- offline and session runners with mock paper routing
- a real paper session runner for IBKR Paper
- formal paper-validation sessions with registry, summaries, and config snapshots
- broker-vs-system reconciliation for orders, fills, and positions
- structured alerts and incidents
- readiness and system-health reporting
- preflight and postflight checks
- conservative recovery and restart assessment
- local scheduler/orchestration plans for safe paper automation
- operational runbooks and archival of session artifacts
- runtime profiles for `development`, `research`, `paper`, and `shadow`
- local runtime bootstrap and service-style `start/stop/restart/status`
- shadow order intents plus shadow-vs-paper and shadow-vs-market reports
- release registry, active release tracking, promotion, rollback, and governance audit trails
- a second historical training-data source based on Polygon Basic for fast bootstrap model training
- economic performance evaluation
- signal quality analysis by score, probability, and buckets
- drift detection for data, prediction outputs, and labels
- run-to-run comparison and economic leaderboard reporting
- automatic Phase 8 reports after paper runs
- execution latency metrics and mock-vs-real comparison artifacts

The active model is visible in `config/active_model.yaml`, not hidden in code.

Important operational note:

- the currently selected models were trained with simulated data and are temporary
- the execution stack does not hardcode those artifacts anywhere
- future retraining should only require registering new artifacts and switching the active model selection
- the execution and evaluation code depends on the active-model interface and normalized decisions, not on a specific model file
- switching from `mock` to `ibkr_paper` should not require changing the order manager, decision engine, or risk engine

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
- resolves one active model from config/registry
- runs inference, decision, and risk on normalized outputs
- routes approved decisions into the paper execution layer through a configurable backend
- persists orders, fills, positions, PnL, and execution state for later audit
- evaluates the resulting economic behavior and signal quality
- tracks drift between current and historical feature / prediction distributions
- writes structured Phase 8 reports for diagnostics and model comparison
- can connect to IBKR Paper for real paper routing while keeping conservative paper-only safety guards
- can validate whether paper operation is stable enough to continue before any future stage

## Project Structure

```text
MicroAlpha-IBKR/
├── app.py
├── config/
│   ├── settings.yaml
│   ├── feature_sets.yaml
│   ├── modeling.yaml
│   ├── active_model.yaml
│   ├── training_data.yaml
│   ├── phase6.yaml
│   ├── phase7.yaml
│   ├── phase8.yaml
│   ├── phase10_11.yaml
│   ├── phase12_14.yaml
│   ├── runtime_profiles.yaml
│   ├── risk.yaml
│   ├── symbols.yaml
│   └── deployment.yaml
├── deployment/
│   └── lan_sync.py
├── data_sources/
│   ├── polygon_client.py
│   ├── polygon_download.py
│   ├── polygon_normalizer.py
│   └── training_dataset_export.py
├── engine/
│   ├── phase6.py
│   └── phase7.py
├── evaluation/
│   ├── io.py
│   ├── performance.py
│   ├── signal_analysis.py
│   ├── execution_metrics.py
│   └── compare_runs.py
├── execution/
│   ├── models.py
│   ├── order_state_machine.py
│   ├── order_manager.py
│   ├── fill_simulator.py
│   ├── paper_broker_mock.py
│   ├── ibkr_paper_backend.py
│   ├── ibkr_state_mapper.py
│   ├── backend.py
│   ├── position_manager.py
│   └── journal.py
├── broker/
│   ├── ib_client.py
│   └── orders.py
├── monitoring/
│   ├── alerts.py
│   ├── drift.py
│   └── paper_monitor.py
├── validation/
│   ├── paper_validation.py
│   ├── readiness.py
│   ├── reconciliation_report.py
│   └── session_tracker.py
├── shadow/
│   ├── comparison.py
│   └── session.py
├── governance/
│   └── releases.py
├── ops/
│   ├── incidents.py
│   ├── orchestrator.py
│   ├── postflight.py
│   ├── preflight.py
│   ├── recovery.py
│   ├── runtime_manager.py
│   ├── runbooks.py
│   └── scheduler.py
├── docs/
│   └── runbooks/
├── scripts/
│   └── deploy/
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
│       ├── phase7/
│       ├── phase8/
│       ├── execution/
│       └── decisions/
├── reporting/
│   ├── performance_report.py
│   ├── trade_report.py
│   └── report_bundle.py
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
│   ├── run_paper_sim_offline.py
│   ├── run_paper_session.py
│   ├── run_paper_session_real.py
│   ├── broker_healthcheck.py
│   ├── execution_status.py
│   ├── show_execution_backend.py
│   ├── evaluate_performance.py
│   ├── analyze_signals.py
│   ├── detect_drift.py
│   ├── compare_runs.py
│   ├── generate_report.py
│   ├── full_evaluation_run.py
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
11. `PC1` runs Phase 6 inference, decision, and risk.
12. `PC1` can run Phase 7 mock paper execution for offline and session validation.
13. `PC1` can run Phase 9 IBKR Paper execution through the same order-management interface.
14. `PC1` can run Phase 10 and 11 paper-validation cycles with monitoring, reconciliation, readiness, and archival.

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

## Phase 5, 6, 7, 8, 9, 10, and 11 Architecture

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

Phase 7 adds:

1. order objects and order state machine
2. backend abstraction for paper execution
3. mock execution backend
4. configurable fill simulator
5. order manager
6. position and portfolio state tracking
7. paper PnL and commission accounting
8. execution journal for orders, fills, positions, and PnL
9. execution status and backend status commands
10. offline and session paper runners that preserve the active-model interface

Phase 8 adds:

1. performance engine with economic metrics per trade and aggregated
2. segmented analysis by score, probability, expected return, symbol, spread, and volume
3. signal quality and calibration diagnostics
4. trade-log analysis across decisions, orders, fills, and reports
5. drift detection for data, model outputs, and offline labels
6. structured report bundles in JSON, CSV, and Parquet
7. run comparison and economic leaderboard updates
8. automatic post-run reporting integrated into the paper execution pipeline

Phase 9 adds:

1. a real `ibkr_paper` execution backend that implements the same backend interface as `mock`
2. broker connectivity, healthcheck, and paper-only safety validation
3. internal-order to broker-order id mapping and reconciliation
4. real broker status mapping into the internal order state machine
5. real fill / execution ingestion with timestamps, commissions, and partial fills
6. execution latency metrics from decision to submit, acknowledgment, and fill
7. decision-vs-execution comparison between expected order intent and actual broker result
8. conservative guard rails to reject live mode, disabled safety flags, unsupported symbols, or missing risk checks

Phase 10 to 14 add:

1. formal paper-validation sessions with `session_id`, summaries, registry, and final state
2. broker reconciliation for orders, fills, and positions
3. structured alerts and incidents with severity and category
4. execution-health and stability metrics across sessions
5. multi-session comparison and validation leaderboards
6. readiness evaluation with `READY`, `REVIEW_NEEDED`, and `NOT_READY`
7. preflight and postflight operational gates
8. conservative recovery logic and safe-restart assessment
9. scheduler/orchestrator planning for local paper automation
10. runbooks plus archival of session artifacts and reports
11. runtime profiles for `development`, `paper`, and `shadow`
12. deployment/runtime bootstrap and service control
13. shadow-mode session execution without order routing
14. model releases, promotion, rollback, and governance reporting

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

### Bootstrap Historical Training Data via Polygon Basic

```bash
python app.py fetch-training-data --provider polygon --symbol SPY --start-date 2025-01-01 --end-date 2025-03-31 --interval 1m --output-path data/training/polygon/SPY_1m_training.csv
python app.py normalize-training-data --provider polygon --input-path path/to/raw_polygon.csv --output-path data/training/polygon/SPY_manual_training.csv
python app.py prepare-training-data --provider polygon --symbol SPY --start-date 2025-01-01 --end-date 2025-03-31 --interval 1m --output-path data/training/polygon/SPY_1m_training.csv
python app.py train --model-type baseline --data-path data/training/polygon/SPY_1m_training.csv
python app.py train --model-type deep --data-path data/training/polygon/SPY_1m_training.csv
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

### Phase 7 Paper Execution Commands

```bash
python app.py show-execution-backend
python app.py execution-status
python app.py run-paper-sim-offline --symbols SPY --limit 200
python app.py run-paper-session --symbols SPY --latest-per-symbol 2
```

### Phase 9 IBKR Paper Commands

```bash
python app.py broker-healthcheck
python app.py show-execution-backend
python app.py show-active-model
python app.py run-paper-session-real --symbols SPY --latest-per-symbol 1
python app.py execution-status --limit 5
```

### Phase 12 to 14 Deployment / Shadow / Governance Commands

```bash
python app.py start-runtime --profile paper
python app.py stop-runtime
python app.py restart-runtime --profile shadow
python app.py service-status
python app.py run-shadow-session --symbols SPY --latest-per-symbol 1
python app.py full-runtime-cycle --profile shadow --symbols SPY --latest-per-symbol 1
python app.py list-model-releases
python app.py promote-model --run-id <run_id> --reason "validated release"
python app.py rollback-model --to <release_id_or_run_id> --reason "rollback after review"
python app.py show-active-release
python app.py runtime-status
python app.py governance-status
```

### Phase 10 and 11 Validation / Hardening Commands

```bash
python app.py run-paper-validation-session --symbols SPY --latest-per-symbol 1
python app.py reconcile-broker-state
python app.py monitor-paper-session --iterations 1
python app.py compare-paper-sessions
python app.py generate-readiness-report
python app.py system-health-report
python app.py full-paper-validation-cycle --symbols SPY --latest-per-symbol 1
python app.py preflight-check
python app.py postflight-check
python app.py list-alerts --limit 20
python app.py list-incidents --limit 20
python app.py generate-runbooks
```

### Runtime Profiles

- `development`: local development defaults, no paper submission.
- `research`: analysis/research profile with no runtime order submission.
- `paper`: expects `ibkr_paper`, paper broker mode, and paper submission enabled.
- `shadow`: runs the full inference/decision/risk chain but records intents only and never submits orders.

Profile selection is controlled by `RUNTIME_PROFILE` and the mappings in `config/runtime_profiles.yaml`.

### Deployment On PC2

Use the lightweight deployment helpers:

```bash
scripts/deploy/bootstrap_pc2.sh
scripts/deploy/runtime_healthcheck.sh
```

The runtime bootstrap validates profile coherence, active release presence, report paths, paper-only safety guards, and broker reachability when the profile requires it.

### Shadow Mode

Shadow mode writes:

- `shadow_intents.jsonl`
- `shadow_session_*.csv`
- `shadow_session_*.parquet`
- `shadow_vs_paper_*.csv`
- `shadow_vs_market_*.csv`
- `shadow_alignment_summary_*.json`

It reuses the same active model, decision engine, and risk engine as paper mode, but it does not route orders.

### Model Releases And Governance

Release artifacts are tracked under `data/models/releases/` with:

- `release_registry.json`
- `active_release.json`
- `release_history.csv`
- `promotion_audit.csv`
- `rollback_audit.csv`
- `release_governance_report.json`

Promotion and rollback update the active model selection automatically through the existing Phase 6 active-model interface.

### Phase 8 Evaluation and Monitoring Commands

```bash
python app.py evaluate-performance --summary-path data/reports/phase7/<summary>.json
python app.py analyze-signals --summary-path data/reports/phase7/<summary>.json
python app.py detect-drift --summary-path data/reports/phase7/<summary>.json
python app.py compare-runs
python app.py generate-report --summary-path data/reports/phase7/<summary>.json
python app.py full-evaluation-run --summary-path data/reports/phase7/<summary>.json
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

Phase 7 and Phase 9 use that selection directly. To replace the temporary simulated models later, register the new artifact and change only the active selection. The execution code does not need to be rewritten.

## Phase 6, 7, and 9 Operational Chain

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

`execution/order_manager.py` then converts approved decisions into normalized paper orders and routes them through the configured execution backend. The default backend remains the mock paper backend in `execution/paper_broker_mock.py`, and Phase 9 adds the real paper backend in `execution/ibkr_paper_backend.py`.

The execution stack records:

- order lifecycle transitions
- backend acknowledgements or rejections
- simulated or real fills
- commissions
- position updates
- paper PnL snapshots
- persisted execution state for status inspection
- broker order ids, perm ids, execution ids, and reconciliation events

## Offline and Session Runners

Phase 6 decision review:

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

Phase 7 offline paper simulation:

```bash
python app.py run-paper-sim-offline --symbols SPY --limit 200
```

This command:

1. loads historical features from the active feature set
2. loads the active model from `config/active_model.yaml`
3. runs inference, decision, and risk
4. builds normalized orders for approved decisions
5. routes them through the configured paper/mock backend
6. simulates fills and commissions
7. updates positions and paper PnL
8. writes JSONL journals plus CSV/Parquet/JSON reports under `data/reports/phase7/`

Phase 7 paper session:

```bash
python app.py run-paper-session --symbols SPY --latest-per-symbol 2
```

This command uses the same stack but only over the latest rows per symbol. It is the current skeleton for the future operational paper session.

Phase 9 IBKR Paper session:

```bash
python app.py run-paper-session-real --symbols SPY --latest-per-symbol 1
```

This command:

1. loads the active model from `config/active_model.yaml`
2. checks that the active execution backend is `ibkr_paper`
3. validates conservative safety guards, including `BROKER_MODE=paper`, `SAFE_TO_TRADE=true`, and `ALLOW_SESSION_EXECUTION=true`
4. connects to IBKR Paper
5. runs inference, decision, and risk
6. reuses the existing order manager to send normalized orders to the real paper backend
7. ingests broker acknowledgments, fills, and execution details
8. updates positions, PnL, and reconciliation journals
9. writes Phase 7 and Phase 8 outputs for later comparison against mock runs

Phase 10 and 11 paper validation session:

```bash
python app.py run-paper-validation-session --symbols SPY --latest-per-symbol 1
```

This command:

1. opens a formal validation session with a tracked `session_id`
2. saves config, model, backend, and scheduler snapshots
3. optionally runs preflight checks
4. runs the real paper session against `ibkr_paper`
5. monitors health, latency, drift, and risk-block behavior
6. reconciles broker-vs-system state
7. generates readiness and session summaries
8. writes alerts, incidents, and auditable artifacts under `data/reports/sessions/<session_id>/`

Full validation cycle:

```bash
python app.py full-paper-validation-cycle --symbols SPY --latest-per-symbol 1
```

This command adds:

1. preflight gating
2. postflight validation
3. archival
4. consolidated system-health reporting
5. scheduler-plan snapshots for safe automation review

Phase 8 report generation:

```bash
python app.py generate-report --summary-path data/reports/phase7/paper_session_summary_<timestamp>.json
```

This command:

1. loads the Phase 7 parquet and summary outputs
2. computes per-trade and aggregated economic metrics
3. analyzes signals by score and probability buckets
4. checks data / prediction / label drift against recent history
5. audits orders, fills, and execution reports
6. writes a structured report bundle under `data/reports/phase8/`
7. updates the economic leaderboard for cross-run comparison
8. writes execution metrics and mock-vs-real comparison artifacts when comparable runs exist

## Interpreting Phase 8 Reports

The main report bundle lives under `data/reports/phase8/<phase7_run_label>/` and includes:

- `run_report.json`
- `metrics_report.json`
- `performance_summary.json`
- `signal_quality.json`
- `drift_report.json`
- `trade_analysis.json`
- `execution_metrics.json`
- `mock_vs_real_comparison.json`
- segment tables such as `score_deciles.csv`, `probability_deciles.csv`, `symbol.csv`, `spread_bucket.csv`

Use the reports this way:

- start with `performance_summary.json` to check total PnL, expectancy, win rate, profit factor, and drawdown
- inspect segment tables to see whether better scores or probabilities actually map to better outcomes
- treat monotonic or near-monotonic score buckets as evidence that the ranking signal may contain edge
- review `drift_report.json` before trusting a good or bad session, because distribution shifts can invalidate comparisons
- use `trade_analysis.json` to spot order rejects, execution inconsistencies, or journaling gaps
- compare `run_report.json` files across sessions with `compare-runs` to rank runs by PnL, Sharpe, and drawdown
- use `execution_metrics.json` to inspect fill ratio, rejection rate, cancel rate, slippage, and broker latency
- use `mock_vs_real_comparison.json` to compare similar runs across the `mock` and `ibkr_paper` backends

## Paper Validation Reports

Phase 10 and 11 write per-session artifacts under:

```text
data/reports/sessions/<session_id>/
```

The most important files are:

- `session_summary.json`
- `preflight_check.json`
- `reconciliation_summary.json`
- `alerts_summary.csv`
- `incidents_summary.csv`
- `readiness_report.json`
- `system_health.json`
- `postflight_check.json`
- `config_snapshot.json`
- `active_model_snapshot.json`
- `backend_snapshot.json`
- `scheduler_plan.json`

Use them this way:

- `session_summary.json`: what happened in the session, counts, PnL, final state
- `reconciliation_summary.json`: whether broker and internal state matched
- `alerts_summary.csv`: operational warnings and critical issues
- `incidents_summary.csv`: classified failures and recovery context
- `readiness_report.json`: whether the system is safe to continue in paper
- `system_health.json`: latest monitoring status across broker, model, drift, and execution

Practical edge heuristic:

- positive net PnL alone is not enough
- the system is more credible when expectancy is positive, drawdown is bounded, score buckets improve from low to high, and drift remains contained
- if top buckets do not outperform lower buckets, the model may be predicting noise rather than tradable edge

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
- `BROKER_MODE` is not `paper`
- the real paper backend is selected while `SAFE_TO_TRADE=false`
- the real paper backend is selected while `ALLOW_SESSION_EXECUTION=false`
- the configured symbol is outside the allowed paper-trading universe
- the broker cannot be reached or does not answer the healthcheck
- internal and broker order ids cannot be correlated cleanly
- real fills would leave the local paper position state inconsistent

## Backend Switching

The active execution backend is resolved through `config/phase7.yaml` and can be overridden from the environment with `ACTIVE_EXECUTION_BACKEND`.

Available values:

- `mock`
- `ibkr_paper`

Examples:

```bash
ACTIVE_EXECUTION_BACKEND=mock python app.py show-execution-backend
ACTIVE_EXECUTION_BACKEND=ibkr_paper python app.py broker-healthcheck
```

The backend switch does not change:

- active-model loading
- inference
- decision logic
- risk checks
- order manager
- journals
- Phase 8 reporting

It only changes the execution backend implementation behind the same interface.

## IBKR Paper Configuration

Minimum paper-routing configuration:

```bash
ACTIVE_EXECUTION_BACKEND=ibkr_paper
BROKER_MODE=paper
SAFE_TO_TRADE=true
ALLOW_SESSION_EXECUTION=true
IBKR_PAPER_HOST=127.0.0.1
IBKR_PAPER_PORT=4002
IBKR_PAPER_CLIENT_ID=101
```

Before running `run-paper-session-real`, verify:

1. TWS or IB Gateway is running in paper mode
2. API access is enabled
3. the configured host and port match the paper listener
4. the active model loads correctly
5. `show-execution-backend` reports `broker_mode=paper`
6. `broker-healthcheck` returns `status=ok`

Before running `full-paper-validation-cycle`, also verify:

1. `preflight-check` returns `status=ok`
2. `generate-runbooks` has created `docs/runbooks/`
3. no unresolved critical alerts or incidents remain from the previous session
4. `SAFE_TO_TRADE` and `ALLOW_SESSION_EXECUTION` are enabled intentionally for paper only

## Preflight, Postflight, and Readiness

Preflight checks validate:

- active model loadability
- backend = `ibkr_paper`
- broker mode = `paper`
- broker reachability
- writable report paths
- risk config availability
- symbol validity
- safe-to-trade flags

Postflight checks validate:

- required reports were created
- reconciliation completed
- alerts were summarized
- archival was completed
- safe restart remains acceptable

Readiness can return:

- `READY`
- `REVIEW_NEEDED`
- `NOT_READY`

The engine uses:

- reconciliation severity
- alert count and critical alerts
- drawdown
- drift
- latency
- session failure rate
- connectivity stability
- positions reconciled
- minimum completed sessions

If readiness is `NOT_READY`, do not continue automated paper validation until the blocking items are resolved.

## Limitations

- Live trading remains explicitly unsupported.
- The currently active models are temporary and were trained from simulated data.
- Replacing those models later should only require new artifacts plus an active-model switch, but the new models still need to respect the existing artifact interface.
- Binary classification targets can only express `LONG` or `NO_TRADE` cleanly; they are not a true short target.
- The session runner currently operates on the latest available feature rows, not on a live streaming feature service.
- The mock backend fills synchronously inside the run; the real paper backend still follows orders synchronously inside the current process and is not yet a full asynchronous event service.
- Phase 8 drift thresholds are simple heuristics; PSI and mean-shift warnings are diagnostic, not proof by themselves.
- Small sample runs can generate incomplete trade statistics and weak calibration analysis.
- Current economic metrics can now include real paper fills, but they still depend on broker-paper availability and on limited local reconciliation rather than a persistent execution store.
- Position reconciliation with broker state is basic and should be treated as a guard rail, not as a final production-grade back-office process.
- Scheduler/orchestration is local and conservative; it is not a distributed operations platform.
- Recovery is intentionally limited to safe checks and conservative reconnect attempts, not automatic resubmission.
- Runtime management is local state management, not a full process supervisor.
- Shadow-mode comparisons are only as good as the paper and realized-market data available for the same timestamps.
- Governance checks are conservative but they do not replace manual operational review before a promotion.
- Polygon Basic bootstrap data is OHLCV-oriented. Synthetic `bid`, `ask`, `bid_size`, and `ask_size` are generated when the source does not provide them, and `last` defaults to `close`.
- Polygon bootstrap datasets are intended to unblock early training, not to replace the IBKR operational data path for execution or final operational validation.
- No portfolio optimization or multi-position allocation logic is included yet.
- Optional `xgboost` and `lightgbm` support still depends on those packages being installed.

## Polygon Basic Bootstrap Training Data

Polygon Basic is integrated as a second historical data source so you can start training immediately even if the current IBKR example CSV is not useful yet.

- IBKR remains the operational route for collection, paper execution, reconciliation, and validation.
- Polygon Basic is a bootstrap route for historical training data.
- The integration does not change the execution stack and does not replace IBKR.

### Why This Exists

- It gives you a fast, free path to a first trainable dataset.
- It lets you bootstrap Phase 5 style experimentation before the operational IBKR data store is mature.
- It keeps the provider-specific logic isolated under `data_sources/`.

### Supported Modes

1. API mode:
- Reads `POLYGON_API_KEY` from `.env`
- Downloads historical aggregates from Polygon Basic
- Normalizes them into the canonical training schema
- Writes CSV, optional Parquet, and optional manifest JSON

2. Manual CSV mode:
- Reads a CSV or Parquet downloaded manually from Polygon
- Detects Polygon-style columns such as `t/o/h/l/c/v`
- Normalizes into the same canonical schema
- Writes the same output artifacts as API mode

### Canonical Output Schema

Every exported bootstrap dataset is normalized to a stable schema:

- `timestamp`
- `symbol`
- `open`
- `high`
- `low`
- `close`
- `last`
- `volume`
- `bid`
- `ask`
- `bid_size`
- `ask_size`
- `source`
- `provider`
- `interval`
- `event_type`
- `collected_at`
- `bootstrap_source`
- `market_data_mode`
- `synthetic_bid_ask_flag`
- `synthetic_depth_flag`

### Real vs Synthetic Fields

When Polygon only provides OHLCV:

- real fields usually are `timestamp`, `open`, `high`, `low`, `close`, and `volume`
- `last` defaults to `close` if missing
- `bid` and `ask` are synthetic using:
  `bid = close * (1 - spread_bps / 20000)`
  `ask = close * (1 + spread_bps / 20000)`
- `bid_size` and `ask_size` default to `POLYGON_DEFAULT_DEPTH_SIZE`

These synthetic fields are never hidden:

- `synthetic_bid_ask_flag`
- `synthetic_depth_flag`
- manifest metadata with `real_columns` and `synthetic_columns`

### Output Files

By default the integration writes under `data/training/polygon/`:

- canonical CSV at the `--output-path`
- optional Parquet with the same basename
- optional manifest JSON with dataset metadata and field provenance

### Commands

API mode:

```bash
python app.py fetch-training-data \
  --provider polygon \
  --symbol SPY \
  --start-date 2025-01-01 \
  --end-date 2025-03-31 \
  --interval 1m \
  --output-path data/training/polygon/SPY_1m_training.csv
```

Manual CSV mode:

```bash
python app.py normalize-training-data \
  --provider polygon \
  --input-path path/to/raw_polygon.csv \
  --output-path data/training/polygon/SPY_manual_training.csv
```

Main convenience command:

```bash
python app.py prepare-training-data \
  --provider polygon \
  --symbol SPY \
  --start-date 2025-01-01 \
  --end-date 2025-03-31 \
  --interval 1m \
  --output-path data/training/polygon/SPY_1m_training.csv
```

### Training Immediately After Preparation

Fastest path:

```bash
python app.py prepare-training-data \
  --provider polygon \
  --symbol SPY \
  --start-date 2025-01-01 \
  --end-date 2025-03-31 \
  --interval 1m \
  --output-path data/training/polygon/SPY_1m_training.csv

python app.py train --model-type baseline --data-path data/training/polygon/SPY_1m_training.csv
```

Optional deep model:

```bash
python app.py train --model-type deep --data-path data/training/polygon/SPY_1m_training.csv
```

If you later want to route the data through the full Phase 5 flow, use the bootstrap dataset to seed a normalized research dataset, then continue with the usual feature, label, and experiment commands. The Polygon integration does not bypass or replace that flow; it just gets you to a first trainable file quickly.

## Next Phase

The next phase should build on this by adding:

- a persistent asynchronous broker event loop for paper execution on the operational PC
- stronger reconciliation across process restarts and broker reconnects
- richer cancel/replace and open-order recovery flows
- real-time feature streaming and session orchestration on the server-side PC2 deployment
- promotion rules for switching the active model safely
- more durable incident management and external alert routing
- longer-horizon validation with more session history before considering any future progression
- alert routing outside files, such as email, chat, or dashboard sinks
- tighter validation against real paper fills and real-time feature drift
