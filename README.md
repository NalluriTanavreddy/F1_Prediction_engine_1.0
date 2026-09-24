# F1 Race Winner Prediction Engine

Classical ML (XGBoost/LightGBM) project predicting F1 race winners for the
2022–2026 seasons. Portfolio project, separate from
[BoxBox](https://github.com/NalluriTanavreddy/boxbox) (an F1 data MCP
server) though it adapts BoxBox's Ergast/Jolpica API client code.

Currently implements the **data ingestion and dataset construction layer**
only. Feature engineering, modeling, and training are later stages.

## Regulation eras

F1's 2026 technical regulations (reduced ground effect, 50/50
electric/combustion power unit split, active aero) are a large enough
discontinuity from the 2022–2025 ground-effect era that the dataset carries
an explicit `era` column (`"2022-2025"` / `"2026"`) rather than treating
season as a smooth numeric trend. `season` and `round` stay as separate
numeric columns for recency-weighting later.

## Setup

```
uv sync
```

## Building the dataset

```
uv run python -m f1_predict.build_dataset
```

This pulls qualifying and race results for every completed round from 2022
through the current season from the
[Jolpica](https://api.jolpi.ca/ergast/f1) API (an Ergast-compatible
drop-in replacement — the original ergast.com shut down after 2024).

- Raw API responses are cached to `data/raw/` (gitignored — regenerate by
  re-running the command; re-runs only fetch rounds not already cached).
- The reshaped dataset is written to `data/processed/race_dataset.parquet`,
  one row per (driver, race). Parquet over CSV because it preserves the
  nullable-int (`Int64`) and `category` dtypes used for `finishing_position`
  and `era` — those matter for how downstream feature engineering handles
  DNFs and the era split, and CSV would round-trip them back to
  ambiguous strings/floats.
- A summary (row/race counts per season and era, DNF count, any rounds
  that failed to fetch) prints at the end.

Sprint races are out of scope — the dataset only covers main Grand Prix
qualifying and race sessions.

## Dataset schema

One row per (season, round, driver):

| column | description |
| --- | --- |
| `season`, `round` | numeric, for recency-weighting |
| `era` | `"2022-2025"` or `"2026"` |
| `race_name`, `circuit_id`, `circuit_name`, `country`, `date` | race identifiers |
| `driver_id`, `driver_code`, `driver_name` | driver identifiers |
| `constructor_id`, `constructor_name` | constructor identifiers |
| `quali_position` | qualifying session classification |
| `grid_position` | actual race-start grid slot (post-penalty) |
| `q1_time_s`, `q2_time_s`, `q3_time_s`, `best_quali_time_s` | parsed lap times in seconds |
| `gap_to_pole_s` | `best_quali_time_s` minus the pole sitter's |
| `finishing_position` | nullable — null for DNFs |
| `won` | 1 if `finishing_position == 1` |
| `points`, `status`, `dnf` | race result detail |
| `missing_qualifying_data` | true if the round had no qualifying data at all |
