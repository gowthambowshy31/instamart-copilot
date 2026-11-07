"""Part A, tasks 2-5: audit the raw exports, clean them, store them as CSV and SQLite.
Run from the repository root:  python src/part_a_clean.py"""
import os
import sqlite3

import pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"


def audit_table(name, df):
    print(f"\n--- {name}: shape {df.shape} ---")
    report = pd.DataFrame({"dtype": df.dtypes.astype(str),
                           "missing": df.isna().sum(),
                           "missing_pct": (df.isna().mean() * 100).round(2)})
    print(report.to_string())


def audit(customers, products, orders):
    print("=== AUDIT (raw) ===")
    for name, df in [("customers.csv", customers), ("products.csv", products), ("orders.csv", orders)]:
        audit_table(name, df)
    amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    age_num = pd.to_numeric(customers["age"], errors="coerce")
    figures = {
        "raw orders.csv rows": len(orders),
        "exact duplicate rows": int(orders.duplicated().sum()),
        "rows with status pending": int((orders["status"] == "pending").sum()),
        "distinct order_id": orders["order_id"].nunique(),
        "non-numeric order_amount": int((amount_num.isna() & orders["order_amount"].notna()).sum()),
        "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
        "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
        "blank age": int(customers["age"].isna().sum()),
        "age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
        "distinct raw city spellings": customers["city"].nunique(),
        "distinct non-blank raw gender spellings": customers["gender"].dropna().nunique(),
        "blank gender": int(customers["gender"].isna().sum()),
    }
    print("\n--- audit figures ---")
    for k, v in figures.items():
        print(f"{k}: {v}")


def clean_customers(c):
    print("\n=== CLEAN customers ===")
    c = c.copy()
    before = c["city"].copy()
    c["city"] = c["city"].str.strip().str.title()
    print(f"city: {int((before != c['city']).sum())} values changed; cities now:")
    print(c["city"].value_counts().to_string())

    gender_map = {"male": "male", "m": "male", "female": "female", "f": "female"}
    before = c["gender"].copy()
    c["gender"] = c["gender"].str.strip().str.lower().map(gender_map).fillna("unknown")
    print(f"gender: {int((before.fillna('') != c['gender']).sum())} values changed; genders now:")
    print(c["gender"].value_counts().to_string())

    c["age"] = pd.to_numeric(c["age"], errors="coerce")
    bad = (c["age"] < 10) | (c["age"] > 100)
    print(f"age: {int(bad.sum())} impossible ages set to missing")
    c.loc[bad, "age"] = float("nan")
    n_missing = int(c["age"].isna().sum())
    c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
    print(f"age: {n_missing} missing ages filled with their city median; "
          f"missing now {int(c['age'].isna().sum())}, range {c['age'].min()}-{c['age'].max()}")

    c["signup_date"] = pd.to_datetime(c["signup_date"], format="%Y-%m-%d")
    print("signup_date: converted to datetime")
    return c


def clean_orders(o, products):
    print("\n=== CLEAN orders ===")
    o = o.copy()
    o["order_ts"] = pd.to_datetime(o["order_ts"], format="%Y-%m-%d %H:%M:%S")
    print(f"1. order_ts converted to datetime ({len(o)} rows)")

    n = len(o)
    o = o.drop_duplicates()
    print(f"2. exact duplicates removed: {n - len(o)} rows -> {len(o)} rows")

    n = len(o)
    o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    print(f"3. kept latest record per order_id: {n - len(o)} older rows removed -> {len(o)} rows")

    o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
    o = o.merge(products[["product_id", "price_inr", "category"]], on="product_id", how="left")
    missing = o["order_amount"].isna()
    rebuilt = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
    o.loc[missing, "order_amount"] = rebuilt[missing]
    print(f"4. unparseable order_amount rebuilt from price: {int(missing.sum())} values")

    neg = o["delivery_minutes"] < 0
    o.loc[neg, "delivery_minutes"] = float("nan")
    print(f"5a. negative delivery_minutes set to missing: {int(neg.sum())} values")
    done = o["status"] == "completed"
    cat_median = o[done].groupby("category")["delivery_minutes"].transform("median")
    fill = done & o["delivery_minutes"].isna()
    o.loc[fill, "delivery_minutes"] = cat_median[fill]
    print(f"5b. missing delivery_minutes of completed orders filled with category median: {int(fill.sum())} values")
    print("    category medians:", o[done].groupby("category")["delivery_minutes"].median().to_dict())

    o = o.drop(columns=["price_inr", "category"]).sort_values("order_id").reset_index(drop=True)
    cancelled = o["status"] == "cancelled"
    print(f"\nresult: {len(o)} rows, statuses {o['status'].value_counts().to_dict()}")
    print(f"missing order_amount: {int(o['order_amount'].isna().sum())}; "
          f"missing delivery (completed): {int(o.loc[o.status == 'completed', 'delivery_minutes'].isna().sum())}; "
          f"missing delivery (cancelled): {int(o.loc[cancelled, 'delivery_minutes'].isna().sum())}")
    print(f"sum of completed order_amount: INR {o.loc[o.status == 'completed', 'order_amount'].sum():,.2f}")
    return o


def store(customers, products, orders):
    os.makedirs(CLEAN, exist_ok=True)
    c = customers.assign(signup_date=customers["signup_date"].dt.strftime("%Y-%m-%d"))
    o = orders.assign(order_ts=orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    c.to_csv(f"{CLEAN}/customers.csv", index=False)
    products.to_csv(f"{CLEAN}/products.csv", index=False)
    o.to_csv(f"{CLEAN}/orders.csv", index=False)
    if os.path.exists(DB):
        os.remove(DB)
    with sqlite3.connect(DB) as con:
        c.to_sql("customers", con, index=False)
        products.to_sql("products", con, index=False)
        o.to_sql("orders", con, index=False)
    print(f"\nstored data/clean/*.csv and {DB} (tables customers, products, orders)")


if __name__ == "__main__":
    customers = pd.read_csv(f"{RAW}/customers.csv")
    products = pd.read_csv(f"{RAW}/products.csv")
    # only truly blank cells are missing; otherwise pandas silently turns the text "N/A" into NaN
    orders = pd.read_csv(f"{RAW}/orders.csv", keep_default_na=False, na_values=[""])
    audit(customers, products, orders)
    store(clean_customers(customers), products, clean_orders(orders, products))
