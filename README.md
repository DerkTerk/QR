# Quant Research Project Scaffold – Folder Guide

This document explains what belongs in each folder of your trading research and backtesting project.

## Project Root
Contains high-level project files and meta-info:
- **README.md** → project overview, quickstart instructions.
- **environment.yml** → reproducible conda environment spec (or generate when needed).
- **.env** → secrets (Eikon key, DB creds) — never committed.
- **.gitignore** → ignores large datasets, outputs, and secrets.
- **pyproject.toml** / **setup.cfg** → optional configs for linting, formatting, and tests.

## 1. `configs/`
All YAML/JSON config files that control experiments, data pulls, or backtests.
- Example: `data_eikon.yaml` → which tickers, date range, fields to pull, where to save.
- Example: `sp500_daily.yaml` → backtest config (data paths, factor list, signal params, cost model).
- No code here — just machine-readable config.

## 2. `data/`
All datasets, local copies, and intermediate data.
- `data/raw/` → raw dumps straight from source (Eikon CSV, JSON). No cleaning, preserved for audit.
- `data/interim/` → partially processed data (e.g., corporate actions applied).
- `data/processed/` → canonical analysis-ready datasets (Parquet/Feather).
  - `prices.parquet` (Date × Symbols)
  - `volumes.parquet`
  - `industry_map.parquet`
  - `membership.parquet`
- `data/external/` → static reference data from other sources (e.g., industry codes).

**Rule:** Raw data is immutable; processed data can be rebuilt from raw.

## 3. `models/`
Serialized model objects (ML models, risk models).
- Example: `xgboost_model.pkl`, `sector_risk_model.joblib`.

## 4. `predictions/`
Saved model outputs / signals.
- Example: `signals.parquet` (date × symbol × signal_value)
- Example: `weights.parquet` (date × symbol × target_weight)

## 5. `reports/`
Generated outputs — metrics, plots, and HTML reports.
- `reports/figures/` → PNG/SVG/HTML charts (equity curves, drawdowns, IC histograms).
- Summary CSV/JSON metrics.

## 6. `notebooks/`
Exploratory work and one-off analyses.
- `eda_data_pull.ipynb` → visual check of new Eikon data.
- `factor_ic_exploration.ipynb` → factor correlation test.

**Rule:** If useful and reproducible, refactor code into `src/`.

## 7. `scripts/`
Command-line entrypoints — run data pulls, backtests, or reports.
- `fetch_eikon.py` → reads `configs/data_eikon.yaml`, calls `src/quant_research/data_portal/eikon_loader.py`.
- `run_backtest.py` → reads a backtest config, loads data, calls your backtest engine.

**Rule:** Only glue + CLI parsing here; business logic lives in `src/`.

## 8. `src/`
All reusable source code — the engine of your project.
### `src/quant_research/data_portal/`
- Data access layer.
- `eikon_loader.py` → pull from Refinitiv API.
- `io.py` → read/write Parquet, CSV.
- `membership.py` → index constituent histories.

### `src/quant_research/factors/`
- Factor construction + transforms.
- `momentum.py`, `value.py`
- `neutralization.py` → z-score, industry-demean, beta-neutral.

### `src/quant_research/backtest/`
- Portfolio simulator(s).
- `portfolio.py` → vectorized daily backtest.
- `execution.py` → slippage, cost models.

### `src/quant_research/costs/`
- Fee, slippage, and borrow cost models.

### `src/quant_research/metrics/`
- Performance metrics, IC/IR, drawdowns, attribution.

### `src/quant_research/cv/`
- Cross-validation logic (walk-forward, purged KFold).

### `src/quant_research/utils/`
- Small helpers (config loader, logging, calendar).

## 9. `tests/`
Unit tests for `src/` code.
- `test_data_portal.py` → does `eikon_loader` return correct columns?
- `test_backtest.py` → does `portfolio` simulate correctly for known data?

## Workflow
1. **Config:** define universe & dates in `configs/data_eikon.yaml`.
2. **Run data pull:**  
   ```bash
   python scripts/fetch_eikon.py --config configs/data_eikon.yaml
   ```
   → Saves to `data/raw/` (optional) and `data/processed/` Parquets.
3. **Run backtest:**  
   ```bash
   python scripts/run_backtest.py --config configs/sp500_daily.yaml
   ```
   → Reads from `data/processed/`, runs factors + backtest in `src/`, writes metrics to `reports/` and signals to `predictions/`.

