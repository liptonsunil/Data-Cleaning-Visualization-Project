"""Generate a realistic, deliberately messy retail sales dataset (raw_sales.csv).

Injected problems (so the cleaning step has real work to do):
  - missing values in several columns
  - exact duplicate rows and near-duplicates (same order id)
  - outliers (typo'd quantities, absurd prices)
  - inconsistent text (case, whitespace, spelling variants)
  - dates in mixed formats, plus a few impossible values
  - numbers stored as text ("$1,200", "N/A")
Swap raw_sales.csv for your own file and re-run clean_and_visualize.py;
the column handling is generic where possible.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

categories = {
    "Electronics": (120, 900),
    "Clothing": (15, 120),
    "Home & Kitchen": (20, 250),
    "Books": (8, 40),
    "Sports": (25, 300),
    "Beauty": (6, 80),
}
regions = ["North", "South", "East", "West"]
channels = ["Online", "Store", "Mobile App"]
segments = ["Consumer", "Corporate", "Home Office"]

cat = rng.choice(list(categories), N, p=[0.2, 0.22, 0.18, 0.12, 0.14, 0.14])
price = np.array([rng.uniform(*categories[c]) for c in cat]).round(2)
qty = rng.integers(1, 8, N)
discount = rng.choice([0, 0.05, 0.1, 0.15, 0.2, 0.3], N, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])

dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
# mild seasonality: resample some orders toward Nov/Dec
boost = rng.random(N) < 0.12
dates = dates.where(~boost, pd.to_datetime("2025-11-01") + pd.to_timedelta(rng.integers(0, 60, N), unit="D"))

df = pd.DataFrame(
    {
        "order_id": [f"ORD-{10000 + i}" for i in range(N)],
        "order_date": dates,
        "customer_id": [f"C{rng.integers(1000, 1800)}" for _ in range(N)],
        "customer_age": rng.normal(38, 11, N).round(),
        "segment": rng.choice(segments, N),
        "region": rng.choice(regions, N),
        "channel": rng.choice(channels, N, p=[0.5, 0.3, 0.2]),
        "category": cat,
        "unit_price": price,
        "quantity": qty,
        "discount": discount,
        "rating": rng.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.08, 0.17, 0.35, 0.35]).astype(float),
    }
)

# ---- inject mess -------------------------------------------------------
def pick(frac):
    return rng.choice(N, int(N * frac), replace=False)

# mixed date formats
fmts = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%d-%m-%Y"]
df["order_date"] = [d.strftime(fmts[rng.integers(0, len(fmts))]) for d in df["order_date"]]
for i in pick(0.004):
    df.loc[i, "order_date"] = "31/02/2025"  # impossible date

# text inconsistencies
for i in pick(0.08):
    df.loc[i, "region"] = df.loc[i, "region"].lower()
for i in pick(0.05):
    df.loc[i, "region"] = " " + df.loc[i, "region"] + " "
for i in pick(0.06):
    df.loc[i, "channel"] = df.loc[i, "channel"].upper()
for i in pick(0.03):
    df.loc[i, "category"] = df.loc[i, "category"].replace("Home & Kitchen", "Home and Kitchen").replace("Electronics", "electronics")

# numbers as text
df["unit_price"] = df["unit_price"].astype(object)
for i in pick(0.05):
    df.loc[i, "unit_price"] = f"${df.loc[i, 'unit_price']:,.2f}"
for i in pick(0.01):
    df.loc[i, "unit_price"] = "N/A"

# outliers
for i in pick(0.01):
    df.loc[i, "quantity"] = int(rng.choice([150, 200, 999, -5]))
for i in pick(0.006):
    df.loc[i, "customer_age"] = int(rng.choice([0, 150, 212, -3]))
for i in pick(0.008):
    price_val = df.loc[i, "unit_price"]
    if not isinstance(price_val, str):
        df.loc[i, "unit_price"] = round(float(price_val) * 50, 2)

# missing values
for col, frac in [("customer_age", 0.07), ("rating", 0.18), ("region", 0.03), ("segment", 0.04), ("discount", 0.05), ("quantity", 0.02)]:
    df.loc[pick(frac), col] = np.nan

# duplicates: exact copies + re-keyed near duplicates
dups = df.sample(120, random_state=1)
near = df.sample(40, random_state=2).copy()
near["rating"] = rng.choice([3.0, 4.0, 5.0], len(near))
df = pd.concat([df, dups, near], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "raw_sales.csv")
df.to_csv(out, index=False)
print(f"wrote {out} with {len(df)} rows, {df.shape[1]} columns")
