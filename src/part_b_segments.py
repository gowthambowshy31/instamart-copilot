"""Part B, tasks 1-3: RFM table, K-means segments, PCA.
Run from the repository root:  python src/part_b_segments.py"""
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4
RFM = ["recency", "frequency", "monetary"]


def build_rfm():
    o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    done = o[o.status == "completed"]
    rfm = done.groupby("customer_id").agg(last_order=("order_ts", "max"),
                                          frequency=("order_id", "nunique"),
                                          monetary=("order_amount", "sum"))
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm[RFM].reset_index()


def name_segments(profile):
    """Business names from each segment's averages. Cluster numbers are only labels."""
    names, left = {}, list(profile.index)
    premium = profile["monetary"].idxmax()
    names[premium] = "Premium Big Spenders"; left.remove(premium)
    loyal = profile.loc[left, "frequency"].idxmax()
    names[loyal] = "Loyal Regulars"; left.remove(loyal)
    at_risk = profile.loc[left, "recency"].idxmax()
    names[at_risk] = "Drifting Away"; left.remove(at_risk)
    for s in left:
        names[s] = "Occasional Shoppers"
    return names, premium


ACTIONS = {"Premium Big Spenders": "Give a priority support line and early access to new electronics.",
           "Loyal Regulars": "Offer a monthly grocery subscription with free delivery to lock in the habit.",
           "Drifting Away": "Send a time-limited win-back coupon on their favourite category.",
           "Occasional Shoppers": "Nudge a second weekly order with a small basket-size discount."}

if __name__ == "__main__":
    rfm = build_rfm()
    print(f"=== RFM (snapshot {SNAPSHOT}) ===\ncustomers: {len(rfm)}")
    print(rfm[RFM].describe().round(2).to_string())
    print(f"recency median {rfm.recency.median():.0f}")

    X = StandardScaler().fit_transform(rfm[RFM])
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"K": k, "inertia": round(km.inertia_, 1),
                     "silhouette": round(silhouette_score(X, km.labels_), 3)})
    print("\n=== K-means, K = 2..8 ===")
    print(pd.DataFrame(rows).to_string(index=False))

    km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
    rfm["segment_id"] = km.labels_
    profile = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                            frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
    names, premium = name_segments(profile)
    profile["segment_name"] = profile.index.map(names)
    profile["action"] = profile["segment_name"].map(ACTIONS)
    print(f"\n=== segment profile, K = {FINAL_K} ===")
    print(profile.to_string())

    pca = PCA(n_components=2).fit(X)
    r = pca.explained_variance_ratio_
    print(f"\nPCA explained_variance_ratio_: {r[0]:.3f}, {r[1]:.3f} (total {r.sum():.3f})")

    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"] == premium
    os.makedirs("outputs", exist_ok=True)
    rfm.to_csv("outputs/customer_segments.csv", index=False)
    print(f"\nwrote outputs/customer_segments.csv ({len(rfm)} rows, premium = segment {premium}, "
          f"{int(rfm.is_premium.sum())} customers)")
