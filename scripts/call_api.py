r"""Tiny example client for the prediction API.

Start the server first (separate terminal):
    uvicorn api.main:app --port 8000
Then run:
    python scripts\call_api.py
"""
import httpx

BASE = "http://127.0.0.1:8000"

# 1) health
print("GET /health ->", httpx.get(f"{BASE}/health").json())

# 2) predict from precomputed features
features = {
    "recency_days": 12, "n_orders": 3, "total_spend": 742.5,
    "avg_order_value": 247.5, "n_distinct_products": 28,
}
print("POST /predict ->", httpx.post(f"{BASE}/predict", json=features).json())

# 3) predict from raw purchase history (features computed server-side, training code reused)
history = {
    "cutoff": "2011-06-01",
    "lines": [
        {"InvoiceNo": "536365", "StockCode": "85123A", "InvoiceDate": "2011-05-10",
         "Quantity": 6, "UnitPrice": 2.55},
        {"InvoiceNo": "536400", "StockCode": "71053", "InvoiceDate": "2011-05-22",
         "Quantity": 3, "UnitPrice": 3.39},
    ],
}
print("POST /predict/from-history ->", httpx.post(f"{BASE}/predict/from-history", json=history).json())
