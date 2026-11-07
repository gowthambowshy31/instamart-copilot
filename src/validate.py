"""Part A, task 5: assert the reference checkpoints on the clean data.
Run from the repository root:  python src/validate.py"""
import sqlite3

import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
done, cancelled = o[o.status == "completed"], o[o.status == "cancelled"]

# customers (task 3)
assert c["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                              "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
assert c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert c["age"].isna().sum() == 0
assert c["age"].between(18, 60).all()

# orders (task 4)
assert len(o) == 12309 and o["order_id"].is_unique
assert len(done) == 11110 and len(cancelled) == 1199
assert (o["status"] == "pending").sum() == 0
assert o["order_amount"].isna().sum() == 0
assert done["delivery_minutes"].isna().sum() == 0
assert cancelled["delivery_minutes"].isna().sum() == 1199
assert (o["delivery_minutes"].dropna() >= 0).all()
assert round(done["order_amount"].sum(), 2) == 2998668.80

# database holds the same three tables
with sqlite3.connect("data/instamart.db") as con:
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("customers", "products", "orders")}
assert counts == {"customers": 1200, "products": 40, "orders": 12309}

print("ALL CHECKPOINTS PASSED")
