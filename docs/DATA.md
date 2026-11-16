# Dataset: UCI Online Retail (ID 352)

## Source, licence, provenance
- **Source:** UCI Machine Learning Repository, dataset 352 — https://archive.ics.uci.edu/dataset/352/online+retail
- **Creator:** Daqing Chen, School of Engineering, London South Bank University.
- **Donated:** 2015-11-05. **DOI:** 10.24432/C5BW33.
- **Licence:** Creative Commons Attribution 4.0 International (**CC BY 4.0**) — reuse/adaptation permitted with attribution.
- **Reference paper:** Chen, Sain & Guo (2012), *Data mining for the online retail industry*, Journal of Database Marketing & Customer Strategy Management, 19(3).
- **File:** `Online Retail.xlsx` (single sheet, 23.7 MB), delivered inside `online+retail.zip`.

### Historical context & scope (important, do not over-generalise)
This is transaction data from **one UK-based, registered, non-store online retailer** selling
mostly **all-occasion giftware**, largely to **wholesale/business customers**, over
**2010-12-01 to 2011-12-09**. Findings here describe *this* business in *that* period.
They **do not** automatically transfer to other sectors, other countries (e.g. Zimbabwean
retailers), or the present day. Treat every result as "what the model learned on this
historical UK gift-wholesaler", not a general law of retail.

## Observed schema (verified from the data, 541,909 rows × 8 columns)

| Column | Dtype | Meaning | Notes from inspection |
|---|---|---|---|
| `InvoiceNo` | string | Transaction/order id | 25,900 distinct. `C` prefix = **cancellation** (9,288 rows). 3 rows prefixed `A` (adjustments / bad debt). |
| `StockCode` | string | Product code | Mostly 5-digit, but many **non-product** codes exist (see below). |
| `Description` | string | Product name | 1,454 missing (0.27%). |
| `Quantity` | int64 | Units on the line | min **-80,995**, max **80,995**; **10,624 rows ≤ 0** (returns/cancellations/adjustments). |
| `InvoiceDate` | datetime | Line timestamp | 2010-12-01 08:26 → 2011-12-09 12:50 (373-day span). |
| `UnitPrice` | float64 | Price per unit (£) | min **-11,062**, max 38,970; **2,515 rows = 0**, **2 rows < 0** (adjustments). |
| `CustomerID` | float64 | Customer id | **24.93% missing** (135,080 rows). 4,372 distinct non-null customers. Stored as float because of NaNs. |
| `Country` | string | Customer country | 12 shown; **UK ≈ 91%** (495,478 rows). |

- **Fully duplicate rows:** 5,268.
- **Granularity:** one row = one **invoice line item**. A **distinct order = distinct `InvoiceNo`**.
  One invoice contains many lines. "Number of orders" must count distinct `InvoiceNo`, not rows.

### Non-product `StockCode` values (exclude from "purchases")
`POST` (postage, 1256), `DOT` (dotcom postage, 710), `M`/`m` (manual, 572), `C2` (carriage, 144),
`D` (discount, 77), `S` (samples, 63), `BANK CHARGES` (37), `AMAZONFEE` (34), `CRUK` (charity, 16),
`PADS`, `B` (adjust bad debt), and `gift_0001_*` gift vouchers. These are fees/admin, not product sales.

### Monthly transaction volume (rows)
```
2010-12  42,481   2011-05  37,030   2011-10  60,742
2011-01  35,147   2011-06  36,874   2011-11  84,711   <- holiday peak
2011-02  27,707   2011-07  39,518   2011-12  25,525   <- PARTIAL (ends Dec 9)
2011-03  36,748   2011-08  35,284
2011-04  29,916   2011-09  50,226
```
Both ends are partial months: December 2010 starts on the 1st (full-ish), **December 2011 is truncated at the 9th**. This truncation drives the label cutoff rule below.

## Data-quality decisions (the cleaning contract)
These choices define a **qualifying purchase** and **eligible customer**. They are applied
identically in training and serving (shared preprocessing). Rationale in parentheses.

1. **Drop rows with missing `CustomerID`** (24.93%). We predict *per existing customer*; an
   untracked basket cannot be attributed to a customer, so it cannot form features or labels.
2. **Drop exact duplicate rows** (5,268) — almost certainly double-logged lines.
3. **A purchase line must have `Quantity > 0` and `UnitPrice > 0`.** Removes the 10,624 non-positive
   quantities and 2,517 non-positive prices (returns, cancellations, zero-price admin, adjustments).
4. **Exclude cancellation invoices** (`InvoiceNo` starts with `C`) and the 3 `A` adjustment rows
   from the definition of a purchase. (v1 keeps it simple: we *exclude* returns rather than netting
   them against prior sales. Netting is a documented future refinement.)
5. **Exclude non-product `StockCode`s** (POST, DOT, M, C2, D, S, BANK CHARGES, AMAZONFEE, CRUK, PADS,
   B, gift vouchers). A postage fee is not a product purchase.
6. **Keep all countries** (UK-dominated) for v1; `Country` is context, not yet a feature.
7. **Never use `CustomerID` as a predictive feature** — it is an identifier, not behaviour.

An **order** = one qualifying `InvoiceNo`. **Order value** = Σ(`Quantity` × `UnitPrice`) over its lines.

## Prediction-window design (chosen from the real date range)

Usable cutoff range is bounded by two needs:
- **Feature lookback (90 days)** before cutoff `C` requires `C - 90d ≥ 2010-12-01` → `C ≥ 2011-03-01`.
- **Label horizon (30 days)** fully observed requires `C + 30d ≤ 2011-12-09` → `C ≤ 2011-11-09`.

So **valid cutoffs live in 2011-03-01 … 2011-11-09**. A customer is *eligible at cutoff C* if they
made ≥1 qualifying purchase in `[C-90d, C)`. The **label** = 1 if they make ≥1 qualifying purchase in
`[C, C+30d)`, else 0. Features use **only** data strictly before `C`; the label uses only `[C, C+30d)`.

### v1 split (single chronological snapshot per split — simple, no random splitting)
| Split | Cutoff `C` | Feature window `[C-90d, C)` | Label window `[C, C+30d)` |
|---|---|---|---|
| Train | 2011-06-01 | 2011-03-03 → 2011-05-31 | 2011-06-01 → 2011-06-30 |
| Validation | 2011-08-01 | 2011-05-03 → 2011-07-31 | 2011-08-01 → 2011-08-30 |
| Test | 2011-10-01 | 2011-07-03 → 2011-09-30 | 2011-10-01 → 2011-10-30 |

Chronological order (train < val < test) guarantees that **every training label is already
observed before we make validation predictions**, and likewise before testing — no peeking into
the future. All test label windows end ≤ 2011-10-30 < 2011-12-09, so labels are fully observed.

**Customers recur across splits** (the same shop appears at June, August, and October cutoffs).
That is *correct* here: the task is predicting repeat behaviour of **existing** customers, so the
model is meant to see customers at multiple points in time. The leakage we must prevent is
*temporal* (future info in features), not customer overlap. We therefore **never random-split**
the customer snapshots — that would leak a customer's future into its own past.

*Future refinement (not v1): stack several monthly training cutoffs to enlarge the training set.*

## Reproducing the data
```powershell
python scripts/get_data.py                      # download + extract + cache parquet
# or, if you already have the zip:
python scripts/get_data.py --zip ".\online+retail.zip"
```
Outputs (git-ignored): `data/raw/Online Retail.xlsx`, `data/interim/online_retail.parquet`.
