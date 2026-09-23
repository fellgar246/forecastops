# Fixtures

Tiny checked-in datasets for tests. `retail-demand-v1/sample.csv` follows schema version `retail-demand-v1` and contains only the required columns. The full synthetic history is produced with `make seed-data` and is not checked in.

`retail-demand-v1/validation/` holds a miniature observation table that passes dataset validation, plus one observation file for each blocking rule. Store, SKU, and category dimensions are shared. Those files are checked with an as-of date of `2026-09-30`, a 14-day staleness window, and a 20× extreme-demand multiple.
