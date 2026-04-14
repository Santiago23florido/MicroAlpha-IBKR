# MicroAlpha-IBKR Phase 4 Drive Sync and Retention

## Purpose

The repository is organized around a two-machine setup:

- `PC1`: research, data validation, feature engineering, training, backtesting
- `PC2`: IBKR collection, local persistence, operational monitoring, local retention, Drive sync

Phase 1 organized the repo and CLI.
Phase 2 added the collector on `PC2`.
Phase 3 added the `PC1` data pipeline and feature generation.
Phase 4 adds a safe sync-and-retention layer between `PC2` local storage and a Google Drive Desktop sync folder.

## Core Rule for Phase 4

SQLite must **not** run live from Google Drive.

The supported model is:

1. SQLite runs only on local disk on `PC2`
2. raw parquet and feature parquet are written locally on `PC2`
3. a sync process copies files into a local Google Drive folder
4. SQLite is copied only as a snapshot backup
5. local cleanup deletes files only after destination validation

This phase does **not** implement autonomous trading, final execution, advanced modeling, RL, or cloud-native orchestration.

## Phase 4 Scope

Phase 4 adds:

- validated sync into a local Google Drive Desktop folder
- safe local snapshot backups of SQLite
- configurable retention and cleanup on `PC2`
- dry-run support for sync and cleanup
- sync status reporting
- logging and JSON reports for sync operations

## PC2 Local + Drive Architecture

The intended flow on `PC2` is now:

1. collector writes local raw parquet into `data/raw/market/`
2. later phases may write local features into `data/features/`
3. runtime SQLite stays local, for example under `data/processed/runtime/`
4. `python app.py backup-sqlite` creates a snapshot in `data/meta/sqlite_backups/`
5. `python app.py sync-drive` copies local artifacts into the configured Google Drive Desktop folder
6. copied files are validated locally in the Drive folder
7. `python app.py cleanup-local` deletes only files that are both validated and old enough

The Google Drive Desktop client is responsible for cloud upload. This code validates only the local copy inside the sync folder.

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
│   ├── drive_sync.py
│   ├── retention.py
│   └── sqlite_backup.py
├── data/
│   ├── loader.py
│   ├── cleaning.py
│   ├── raw/
│   │   └── market/
│   ├── features/
│   ├── processed/
│   ├── meta/
│   │   └── sqlite_backups/
│   ├── logs/
│   └── reports/
│       └── sync/
├── ingestion/
│   └── collector.py
├── features/
│   └── feature_pipeline.py
├── labels/
│   └── dataset_builder.py
├── monitoring/
│   ├── healthcheck.py
│   ├── logging.py
│   └── data_quality.py
├── scripts/
│   ├── run_collector.py
│   ├── build_features.py
│   ├── sync_to_drive.py
│   ├── cleanup_local.py
│   ├── backup_sqlite.py
│   └── healthcheck.py
└── tests/
```

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

### Phase 4 Settings

Relevant sync settings:

- `SYNC_ENABLED`
- `GOOGLE_DRIVE_ROOT`
- `GOOGLE_DRIVE_SUBDIR`
- `SYNC_RAW_ENABLED`
- `SYNC_FEATURES_ENABLED`
- `SYNC_SQLITE_ENABLED`
- `SYNC_LOGS_ENABLED`
- `DELETE_AFTER_SYNC`
- `DELETE_MIN_AGE_HOURS`
- `RETENTION_DAYS_LOCAL`
- `SYNC_DRY_RUN`
- `SYNC_VALIDATE_CHECKSUM`
- `SQLITE_BACKUP_FILENAME`
- `SQLITE_SOURCE_PATH`
- `SQLITE_BACKUP_DIR`
- `SYNC_REPORT_DIR`

Inspect the effective config with:

```bash
python app.py show-config
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

## Google Drive Desktop Setup

This implementation assumes you have a local folder already managed by Google Drive Desktop, for example:

```text
G:/My Drive
/mnt/g/My Drive
/home/<user>/Google Drive
```

Set `GOOGLE_DRIVE_ROOT` to that local folder. The application will create or use a subdirectory inside it, controlled by `GOOGLE_DRIVE_SUBDIR`.

Example:

```dotenv
GOOGLE_DRIVE_ROOT=/mnt/g/My Drive
GOOGLE_DRIVE_SUBDIR=microalpha
```

The resulting local Drive base would be:

```text
/mnt/g/My Drive/microalpha
```

## Local Data Layout on PC2

Raw collector output:

```text
data/raw/market/YYYY-MM-DD/SYMBOL/collector_*.parquet
```

Processed feature output:

```text
data/features/YYYY-MM-DD/SYMBOL.parquet
```

SQLite runtime database:

```text
data/processed/runtime/microalpha.db
```

Local SQLite snapshot backups:

```text
data/meta/sqlite_backups/*.sqlite
```

Sync reports:

```text
data/reports/sync/*.json
```

## Drive Destination Layout

Inside the local Google Drive folder, files are copied into this structure:

```text
<GOOGLE_DRIVE_ROOT>/<GOOGLE_DRIVE_SUBDIR>/
├── raw/
│   └── market/YYYY-MM-DD/SYMBOL/collector_*.parquet
├── features/
│   └── YYYY-MM-DD/SYMBOL.parquet
├── meta/
│   └── sqlite/*.sqlite
└── logs/
```

## Safety Model

Before any local file is considered synced, the system validates:

- destination file exists
- destination size is greater than zero
- destination size matches the source exactly
- optional checksum match when enabled

Files are never deleted without validation.

## Main Commands

### Collector on PC2

```bash
python app.py --environment deploy collect --once
python app.py --environment deploy collect --max-cycles 120
```

### Feature Build on PC1

```bash
python app.py --environment development build-features
python app.py --environment development build-features --symbols SPY QQQ
```

### Sync to Google Drive

Dry-run using configured defaults:

```bash
python app.py --environment deploy sync-drive
```

Explicit execution:

```bash
python app.py --environment deploy sync-drive --no-dry-run
```

Sync only selected categories:

```bash
python app.py --environment deploy sync-drive --categories raw features
```

Enable deletion right after validated sync:

```bash
python app.py --environment deploy sync-drive --no-dry-run --delete-after-sync
```

Script entrypoint:

```bash
python scripts/sync_to_drive.py --environment deploy --no-dry-run
```

### Local Cleanup

Dry-run cleanup:

```bash
python app.py --environment deploy cleanup-local
```

Execute cleanup:

```bash
python app.py --environment deploy cleanup-local --no-dry-run
```

Script entrypoint:

```bash
python scripts/cleanup_local.py --environment deploy --no-dry-run
```

### SQLite Backup

Dry-run backup:

```bash
python app.py --environment deploy backup-sqlite
```

Create a real local snapshot:

```bash
python app.py --environment deploy backup-sqlite --no-dry-run
```

Script entrypoint:

```bash
python scripts/backup_sqlite.py --environment deploy --no-dry-run
```

### Sync Status

```bash
python app.py --environment deploy sync-status
```

This reports:

- pending local files
- already-synced files
- invalid destination copies
- estimated deletable bytes
- Drive folder availability
- latest sync report path and status

### Healthcheck

```bash
python app.py healthcheck --skip-broker
python app.py --environment deploy healthcheck
```

## Phase 3 Data Pipeline Still Present

Phase 4 does not remove the research pipeline introduced in phase 3. The repository still supports:

- raw parquet loading
- data-quality validation
- deterministic cleaning
- ORB, microstructure, intraday, and cost features
- dataset preparation for later training

## Logging and Reports

Shared logging is handled by `monitoring/logging.py`.

Phase 4 logs:

- sync start and finish
- Drive path availability
- SQLite backup creation
- files copied
- files skipped
- validation failures
- files deleted
- dry-run mode

Structured JSON reports are written under:

```text
data/reports/sync/
```

## Important Limitation

Validation only confirms the file exists in the local Google Drive folder and matches the local source by size and, optionally, checksum.

It does **not** confirm that Google Drive has already uploaded the file to the cloud. That responsibility belongs to the Google Drive Desktop client.

## Current Limitations

Phase 4 is intentionally conservative:

- no cloud API integration
- no remote upload acknowledgment beyond local Drive-folder validation
- no deduplicated manifest database for sync state
- no advanced space-pressure scheduler yet
- no object-store or remote checksum catalog
- no final deployment supervisor service yet

## What Phase 5 Should Add

Phase 5 can now build on this foundation:

- scheduled sync and cleanup jobs on `PC2`
- stronger retention policies by class of data
- Drive sync monitoring and alerting
- artifact promotion from Drive into `PC1` research workflows
- tighter end-to-end automation between collection, sync, feature build, and training

## Tests

Run the full suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests
```

Phase 4 specifically adds tests for:

- SQLite snapshot backup creation
- Drive sync dry-run and real copy behavior
- validation before deletion
- cleanup of only confirmed synced files
- sync status reporting
