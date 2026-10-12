"""Clean raw_sales.csv and build a visual report.

Usage:  python clean_and_visualize.py [raw_csv] [output_dir]
Outputs (in output_dir, default ./output):
  cleaned_sales.csv        - analysis-ready data
  cleaning_log.csv         - every cleaning action with row counts
  cleaning_impact.png      - before/after evidence for the cleaning
  dashboard.png            - the insight dashboard
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "raw_sales.csv")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "output")
OUT.mkdir(parents=True, exist_ok=True)

log = []


def note(step, detail, rows_affected):
    log.append({"step": step, "detail": detail, "rows_affected": int(rows_affected)})
    print(f"[{step}] {detail}: {int(rows_affected)}")


# ---------------------------------------------------------------- 1. load
# read everything as text; only empty cells are NaN, so junk like "N/A" is caught
# (and counted) by the numeric conversion below
raw = pd.read_csv(RAW, dtype=str, keep_default_na=False, na_values=[""])
df = raw.copy()
note("load", f"rows x cols = {df.shape[0]} x {df.shape[1]}", df.shape[0])

# ------------------------------------------------- 2. standardise text
text_cols = ["segment", "region", "channel", "category"]
for c in text_cols:
    before = df[c].copy()
    df[c] = df[c].str.strip().str.title()
    note("text", f"trimmed/title-cased '{c}'", (before.fillna("") != df[c].fillna("")).sum())
changed = (df["category"] == "Home And Kitchen").sum()
df["category"] = df["category"].replace({"Home And Kitchen": "Home & Kitchen"})
note("text", "merged 'Home and Kitchen' spelling variant", changed)

# ------------------------------------------ 3. convert numeric columns
def to_number(s):
    s = s.str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce").astype("float64")  # "N/A" and junk become NaN


for c in ["unit_price", "quantity", "discount", "customer_age", "rating"]:
    before_na = df[c].isna().sum()
    df[c] = to_number(df[c].astype("string"))
    note("numeric", f"converted '{c}' to numbers (new NaNs from junk text)", df[c].isna().sum() - before_na)

# ----------------------------------------------------- 4. parse dates
fmts = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%d-%m-%Y"]
parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
for f in fmts:
    mask = parsed.isna()
    parsed[mask] = pd.to_datetime(df.loc[mask, "order_date"], format=f, errors="coerce")
bad_dates = parsed.isna().sum()
df["order_date"] = parsed
df = df.dropna(subset=["order_date"])
note("dates", "dropped rows with unparseable/impossible dates", bad_dates)

# -------------------------------------------------- 5. duplicates
n = len(df)
df = df.drop_duplicates()
note("duplicates", "removed exact duplicate rows", n - len(df))
n = len(df)
df = df.drop_duplicates(subset="order_id", keep="first")
note("duplicates", "removed repeated order_id (near-duplicates)", n - len(df))

# ---------------------------------- 6. impossible values -> missing
bad_qty = (df["quantity"] <= 0).sum()
df.loc[df["quantity"] <= 0, "quantity"] = np.nan
note("invalid", "quantity <= 0 set to missing", bad_qty)
bad_age = ((df["customer_age"] < 16) | (df["customer_age"] > 100)).sum()
df.loc[(df["customer_age"] < 16) | (df["customer_age"] > 100), "customer_age"] = np.nan
note("invalid", "customer_age outside 16-100 set to missing", bad_age)

# snapshot for before/after chart
price_before = df["unit_price"].copy()
qty_before = df["quantity"].copy()
missing_before = df.isna().sum()


# ------------------------------------------------ 7. outliers (IQR)
def iqr_cap(series, group=None, k=1.5):
    """Winsorise values beyond Q1-k*IQR / Q3+k*IQR (per group if given)."""
    def cap(s):
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return s.clip(q1 - k * iqr, q3 + k * iqr)
    return series.groupby(group).transform(cap) if group is not None else cap(series)


capped = iqr_cap(df["unit_price"], df["category"])
note("outliers", "unit_price capped at 1.5xIQR within category", (capped != df["unit_price"]).sum() - df["unit_price"].isna().sum())
df["unit_price"] = capped
capped = iqr_cap(df["quantity"])
note("outliers", "quantity capped at 1.5xIQR", (capped != df["quantity"]).sum() - df["quantity"].isna().sum())
df["quantity"] = capped

# -------------------------------------------------- 8. missing values
df["rating_missing"] = df["rating"].isna()  # keep the fact that it was missing
for c in ["segment", "region"]:
    note("missing", f"'{c}' -> 'Unknown'", df[c].isna().sum())
    df[c] = df[c].fillna("Unknown")
note("missing", "discount -> 0 (assumed no discount)", df["discount"].isna().sum())
df["discount"] = df["discount"].fillna(0)
for c, grp in [("unit_price", "category"), ("quantity", "category"), ("customer_age", "segment")]:
    note("missing", f"'{c}' -> median by {grp}", df[c].isna().sum())
    df[c] = df[c].fillna(df.groupby(grp)[c].transform("median")).fillna(df[c].median())
df["quantity"] = df["quantity"].round().astype(int)
df["customer_age"] = df["customer_age"].round().astype(int)
# rating is left missing on purpose: imputing opinions would invent data
note("missing", "rating left missing (flagged in rating_missing)", df["rating"].isna().sum())

# ------------------------------------------------ 9. derived columns
df["revenue"] = (df["unit_price"] * df["quantity"] * (1 - df["discount"])).astype("float64").round(2)
df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
df = df.sort_values("order_date").reset_index(drop=True)
df.to_csv(OUT / "cleaned_sales.csv", index=False)
pd.DataFrame(log).to_csv(OUT / "cleaning_log.csv", index=False)
note("done", f"cleaned rows kept (of {len(raw)} raw)", len(df))

# ================================================================ visuals
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
money = lambda x, _: f"${x/1000:,.0f}k" if x >= 1000 else f"${x:,.0f}"
fmt = plt.FuncFormatter(money)

# ---------- figure 1: cleaning impact
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
ax = axes[0]
mb = missing_before[missing_before > 0].sort_values()
ma = df.isna().sum().reindex(mb.index).fillna(0)
y = np.arange(len(mb))
ax.barh(y + 0.2, mb.values, 0.38, color=ORANGE, label="Before")
ax.barh(y - 0.2, ma.values, 0.38, color=BLUE, label="After")
ax.set_yticks(y, mb.index)
ax.set_title("Missing values: filled except rating")
ax.set_xlabel("Rows with a missing value")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", visible=False)

for ax, b, a, title, lim in [
    (axes[1], price_before, df["unit_price"], "Unit price: extreme values tamed", None),
    (axes[2], qty_before, df["quantity"], "Quantity: typos (150, 999) removed", None),
]:
    data = pd.DataFrame({"value": pd.concat([b.dropna(), a]),
                         "stage": ["Before"] * b.notna().sum() + ["After"] * len(a)})
    sns.boxplot(data=data, x="stage", y="value", hue="stage", palette={"Before": ORANGE, "After": BLUE},
                width=0.5, fliersize=2, linewidth=1, ax=ax, legend=False,
                flierprops={"markerfacecolor": MUTED, "markeredgecolor": "none", "alpha": 0.5})
    ax.set_title(title)
    ax.set_yscale("log")  # outliers are orders of magnitude away; log keeps both boxes readable
    ax.set_xlabel("")
    ax.set_ylabel("Log scale")
    ax.grid(axis="x", visible=False)
fig.suptitle(f"Cleaning impact: {len(raw):,} raw rows -> {len(df):,} clean rows", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.savefig(OUT / "cleaning_impact.png", dpi=160)
plt.close(fig)

# ---------- figure 2: dashboard
monthly = df.groupby("month")["revenue"].sum()
cat_rev = df.groupby("category")["revenue"].sum().sort_values()
reg_chan = df.pivot_table(index="region", columns="channel", values="revenue", aggfunc="sum")
reg_chan = reg_chan.drop(index="Unknown", errors="ignore")
heat = df.pivot_table(index="category", columns="region", values="revenue", aggfunc="sum").drop(columns="Unknown", errors="ignore")
disc = df.groupby("discount").agg(rev_per_order=("revenue", "mean"), orders=("order_id", "count"))
rating_cat = df.groupby("category")["rating"].mean().sort_values()

top_cat = cat_rev.idxmax()
peak_month = monthly.idxmax()
total = df["revenue"].sum()

fig = plt.figure(figsize=(16, 11), constrained_layout=True)
gs = fig.add_gridspec(3, 6, height_ratios=[0.5, 1.5, 1.5])

# KPI tiles
kpis = [
    ("Total revenue", f"${total/1e6:,.2f}M"),
    ("Orders", f"{len(df):,}"),
    ("Avg order value", f"${df['revenue'].mean():,.0f}"),
    ("Avg rating", f"{df['rating'].mean():.2f} / 5"),
    ("Top category", top_cat),
    ("Peak month", peak_month.strftime("%b %Y")),
]
for i, (label, value) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.axis("off")
    ax.text(0, 0.62, value, fontsize=21, fontweight="bold", va="center")
    ax.text(0, 0.18, label, fontsize=10.5, color=MUTED, va="center")

# monthly revenue
ax = fig.add_subplot(gs[1, 0:3])
ax.plot(monthly.index, monthly.values, color=BLUE, lw=2.2, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax.yaxis.set_major_formatter(fmt)
ax.set_title(f"Revenue peaked in {peak_month.strftime('%B')} at {money(monthly.max(), None)}")
ax.annotate(money(monthly.max(), None), (peak_month, monthly.max()), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10)
ax.set_ylim(0, monthly.max() * 1.15)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
ax.grid(axis="x", visible=False)

# revenue by category
ax = fig.add_subplot(gs[1, 3:6])
share = cat_rev.iloc[-1] / cat_rev.sum()
ax.barh(cat_rev.index, cat_rev.values, color=BLUE, height=0.62)
for i, v in enumerate(cat_rev.values):
    ax.text(v, i, f" {money(v, None)}", va="center", fontsize=9.5)
ax.set_title(f"{top_cat} leads with {share:.0%} of revenue")
ax.xaxis.set_major_formatter(fmt)
ax.set_xlim(0, cat_rev.max() * 1.15)
ax.grid(axis="y", visible=False)

# region x channel (grouped bars, 3 series)
ax = fig.add_subplot(gs[2, 0:2])
w = 0.26
for j, (ch, col) in enumerate(zip(reg_chan.columns, [BLUE, ORANGE, AQUA])):
    ax.bar(np.arange(len(reg_chan)) + (j - 1) * (w + 0.02), reg_chan[ch], w, color=col, label=ch)
ax.set_xticks(np.arange(len(reg_chan)), reg_chan.index)
ax.yaxis.set_major_formatter(fmt)
ax.set_title("Online is the biggest channel in every region")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.grid(axis="x", visible=False)

# heatmap category x region
ax = fig.add_subplot(gs[2, 2:4])
sns.heatmap(heat / 1000, annot=True, fmt=".0f", cmap=sns.light_palette(BLUE, as_cmap=True), cbar=False,
            linewidths=2, linecolor=SURFACE, ax=ax, annot_kws={"fontsize": 9})
ax.set_title("Revenue ($k) by category and region")
ax.set_xlabel("")
ax.set_ylabel("")
ax.grid(False)

# discount vs revenue per order
ax = fig.add_subplot(gs[2, 4:6])
ax.bar([f"{int(d*100)}%" for d in disc.index], disc["rev_per_order"], color=BLUE, width=0.6)
for i, v in enumerate(disc["rev_per_order"]):
    ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
deep = df.loc[df["discount"] >= 0.2, "revenue"].mean()
shallow = df.loc[df["discount"] < 0.2, "revenue"].mean()
ax.set_title(f"Orders with 20%+ discounts average ${deep:,.0f} vs ${shallow:,.0f}")
ax.set_xlabel("Discount level")
ax.yaxis.set_major_formatter(fmt)
ax.grid(axis="x", visible=False)

fig.suptitle("Retail Sales 2025 - Insights Dashboard", x=0.005, ha="left", fontsize=17, fontweight="bold")
fig.savefig(OUT / "dashboard.png", dpi=150)
plt.close(fig)

print("\nSaved:", ", ".join(p.name for p in sorted(OUT.iterdir())))
print(f"Key facts: top category={top_cat}, peak month={peak_month:%Y-%m}, total=${total:,.0f}")
