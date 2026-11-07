"""Part A, tasks 6-7: run the seven SQL queries, cross-check in pandas, describe the data.
Run from the repository root:  python src/part_a_analysis.py"""
import re
import sqlite3

import pandas as pd

pd.set_option("display.width", 140)


def load_queries(path="sql/queries.sql"):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_sql():
    con = sqlite3.connect("data/instamart.db")
    q = {name: pd.read_sql_query(sql, con) for name, sql in load_queries().items()}
    con.close()
    print("=== SQL (sql/queries.sql) ===")
    print(f"Q1 customers with > INR 8000 in completed orders above INR 500: {len(q['Q1'])}")
    print(q["Q1"].head(5).to_string(index=False))
    top = q["Q2"].iloc[0]
    print(f"\nQ2 rows: {len(q['Q2'])}; total revenue INR {q['Q2']['revenue'].sum():,.2f}; "
          f"largest cell {top.city} / {top.category} INR {top.revenue:,.2f}")
    print(f"\nQ3 products above average price: {len(q['Q3'])}")
    print(f"Q4 customers with no completed order: {len(q['Q4'])}")
    print(f"Q5 customers above average spend: {len(q['Q5'])}; average INR {q['Q5']['avg_spend'].iloc[0]:,.2f}")
    q6 = q["Q6"]
    print("\nQ6 top customer per city:")
    print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
    diff = q6[q6.rnk != q6.row_num]
    print(f"Q6 rows where RANK() != ROW_NUMBER(): {len(diff)}")
    print(diff.to_string(index=False))
    print(f"\nQ7 daily rows: {len(q['Q7'])}; last cumulative INR {q['Q7']['cumulative_revenue'].iloc[-1]:,.2f}")
    return q


def pandas_checks(q2):
    o = pd.read_csv("data/clean/orders.csv")
    c = pd.read_csv("data/clean/customers.csv")
    p = pd.read_csv("data/clean/products.csv")
    done = o[o.status == "completed"].merge(c, on="customer_id").merge(p, on="product_id")

    print("\n=== pandas cross-check ===")
    pivot = pd.pivot_table(done, values="order_amount", index="city", columns="category",
                           aggfunc="sum", margins=True, margins_name="Total")
    print(pivot.round(2).to_string())
    print(f"pivot grand total INR {pivot.loc['Total', 'Total']:,.2f} vs SQL Q2 total INR {q2['revenue'].sum():,.2f}"
          f" -> equal: {round(pivot.loc['Total', 'Total'], 2) == round(q2['revenue'].sum(), 2)}")

    per_city = done.groupby("city").agg(sum_inr=("order_amount", "sum"), mean_inr=("order_amount", "mean"),
                                        orders=("order_amount", "count")).round(2)
    print("\nper city (completed orders):")
    print(per_city.to_string())

    amt = done["order_amount"]
    q1, q3 = amt.quantile(0.25), amt.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print("\n=== completed order_amount statistics ===")
    print(f"mean {amt.mean():.2f}  median {amt.median():.2f}  skewness {amt.skew():.2f}")
    print(f"Q1 {q1:.2f}  Q2 {amt.median():.2f}  Q3 {q3:.2f}  IQR {iqr:.2f}")
    print(f"lower fence {low:.2f} -> {int((amt < low).sum())} outliers below; "
          f"upper fence {high:.2f} -> {int((amt > high).sum())} outliers above")
    print(f"min {amt.min():.2f}  max {amt.max():.2f}")
    print("\norders and revenue by category:")
    print(done.groupby("category")["order_amount"].agg(["count", "sum", "mean"]).round(2).to_string())

    corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    print("\ncorrelation matrix (completed orders):")
    print(corr.round(3).to_string())
    pairs = corr.where(~pd.DataFrame([[i >= j for j in range(4)] for i in range(4)],
                                     index=corr.index, columns=corr.columns)).stack()
    best = pairs.abs().idxmax()
    print(f"largest |correlation|: {best[0]} vs {best[1]} = {pairs[best]:.3f}")


if __name__ == "__main__":
    queries = run_sql()
    pandas_checks(queries["Q2"])
