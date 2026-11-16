# Retail Repeat-Purchase Prediction Service

Predict whether an **existing customer will make another purchase within the next 30 days**,
using only information available at prediction time. Built on the UCI *Online Retail* dataset
for the DataTalksClub Machine Learning Zoomcamp midterm project.

> **Status:** work in progress. Milestone 1 (project setup, data acquisition, data inspection)
> is complete. See `docs/DATA.md` for the verified dataset schema, quality issues, and the
> leakage-safe prediction-window design. Later milestones (features, models, API, Docker) follow.

## Problem

A UK-based online gift wholesaler wants to know **which of its current customers are likely to
buy again soon**, so limited retention attention can be focused. We frame this as **binary
classification** at a historical *prediction cutoff*:

- **Unit of observation:** one customer, observed at a cutoff date `C`.
- **Features:** behaviour in the 90 days *before* `C` (recency, order count, spend, basket variety).
- **Target:** `1` if the customer makes ≥1 qualifying purchase in `[C, C+30 days)`, else `0`.

Predicting a repeat purchase is **not** proof that a marketing action *causes* a purchase or
lifts revenue — it ranks likelihood, it does not establish causation. Results describe this
specific historical UK business and do **not** generalise to other markets (e.g. Zimbabwe).

## Course alignment (DataTalksClub ML Zoomcamp)

Verified against the official evaluation rubric (projects folder, 16 points + peer review):
Problem description, EDA, Model training (linear **and** tree-based, tuned), training script,
reproducibility, deployment (web service), dependency/env management, containerization, cloud
deployment. This project targets each of those criteria. The verified rubric is summarised in
`docs/COURSE.md`.

## Repository layout
```
data/            raw/ interim/ processed/   (git-ignored; regenerate with scripts/get_data.py)
scripts/         get_data.py                 reproducible download + extract + parquet cache
notebooks/       exploration & explanation   (reusable logic lives in src/, not duplicated here)
src/retail_repeat/  data loading, cleaning, feature/label building, training   (added next)
models/          saved preprocessing + model artifacts (git-ignored)
api/             FastAPI prediction service   (added at deployment milestone)
tests/           leakage, feature, input-validation, serialization, API tests
docs/            DATA.md (dataset + design), COURSE.md (rubric)
reports/         metrics and figures
```

## Quickstart (Windows PowerShell)
```powershell
# 1) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) install dependencies
pip install -r requirements.txt

# 3) get the data (≈24 MB download, or use your local zip)
python scripts\get_data.py --zip ".\online+retail.zip"

# 4) build the labelled customer tables (train/val/test)
python scripts\build_datasets.py

# 5) train the final model -> models\model.joblib
python scripts\train.py

# 6) (optional) reproduce the model comparison and the honest test evaluation
python scripts\run_baselines.py
python scripts\train_models.py
python scripts\evaluate_test.py          # writes reports\test_metrics.json

# run the tests
python -m pytest -q
```

## Prediction API

Start the service:
```powershell
uvicorn api.main:app --port 8000
```
Interactive docs: http://127.0.0.1:8000/docs

**Endpoints**
- `GET /health` → `{status, model_loaded, model_version}`
- `POST /predict` → score **precomputed** features, returns `{probability, will_repeat, threshold, model_version}`
- `POST /predict/from-history` → send a customer's **raw purchase lines** + a `cutoff`; the service computes
  features with the *same* code as training and scores them.

**Feature contract for `/predict`.** The five features must be produced by the project's own code,
`retail_repeat.features.compute_features_at_cutoff`, over a **90-day lookback** ending at the prediction
cutoff, counting only *qualifying purchases* (see `docs/DATA.md`):

| field | meaning | constraint |
|---|---|---|
| `recency_days` | days since last qualifying purchase | `0 ≤ x ≤ 90` |
| `n_orders` | distinct invoices in the lookback | `≥ 1` |
| `total_spend` | Σ(quantity × unit price) | `> 0` |
| `avg_order_value` | `total_spend / n_orders` | `> 0` |
| `n_distinct_products` | distinct product codes | `≥ 1` |

Example (PowerShell):
```powershell
$f = @{ recency_days=12; n_orders=3; total_spend=742.5; avg_order_value=247.5; n_distinct_products=28 }
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" -Body ($f | ConvertTo-Json)
```
A ready-made client is in `scripts\call_api.py` (`python scripts\call_api.py`).

> Model performance is an honest estimate from the untouched test period (Oct 2011):
> **ROC-AUC 0.668, average precision 0.575, F1 0.510** — clearly above the recency baseline
> (0.617 / 0.405 / 0.503). This ranks repurchase likelihood; it does not prove causation, and it
> reflects a historical UK gift-wholesaler only. Full metrics in `reports\test_metrics.json`.

## Docker

The service is containerized. The image bundles the app code and the **pre-trained**
`models/model.joblib`, so a reviewer needs only Docker — no Python setup.

```powershell
# prerequisite (once): build data + train the model so models\model.joblib exists
python scripts\build_datasets.py
python scripts\train.py

# build the image
docker build -t retail-repeat:latest .

# run it (maps container port 8000 -> host 8000)
docker run --rm -p 8000:8000 retail-repeat:latest
```
Then open http://127.0.0.1:8000/docs, or:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```
> If host port 8000 is already in use, map a different one, e.g. `-p 8001:8000`, and use
> `http://127.0.0.1:8001`. The container always listens on 8000 internally.
> First request may 502/empty for ~20s while the app boots — the container's `HEALTHCHECK`
> flips to `healthy` once it's ready (`docker ps` shows the status).

Notes:
- Runtime dependencies are pinned in `requirements-api.txt` to match the versions the model was
  pickled with (so `joblib.load` is warning-free). If you retrain with newer libraries, update those pins.
- The image runs as a non-root user and defines a `HEALTHCHECK` against `/health`.
- `.dockerignore` keeps data, notebooks, tests, and reports out of the image.

## Data

UCI *Online Retail* (ID 352), CC BY 4.0, Chen (2015), DOI 10.24432/C5BW33.
541,909 invoice lines, 2010-12-01 → 2011-12-09, ~4,372 identifiable customers.
Full schema, quality issues, cleaning contract, and the train/validation/test cutoff design:
**see `docs/DATA.md`.**
