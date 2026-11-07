"""Part B, tasks 4-8: lapse label, leakage-free features, tuned models, balancing, saving, retention list.
Run from the repository root (after part_b_segments.py):  python src/part_b_lapse.py"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF, SEPT_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")
NUMERIC = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
NOMINAL = ["city", "gender", "fav_category"]


def build_dataset():
    o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    p = pd.read_csv("data/clean/products.csv")
    o = o.merge(p[["product_id", "category"]], on="product_id")

    before = o[o.order_ts < CUTOFF]                       # the only orders features may use
    done = before[before.status == "completed"]
    f = done.groupby("customer_id").agg(last_order=("order_ts", "max"), frequency=("order_id", "nunique"),
                                        monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
                                        avg_delivery=("delivery_minutes", "mean"))
    f["recency"] = (CUTOFF - f.pop("last_order")).dt.days
    f["fav_category"] = done.groupby("customer_id")["category"].agg(lambda s: s.value_counts().sort_index()
                                                                    .sort_values(ascending=False, kind="stable").index[0])
    status_counts = before.groupby("customer_id")["status"].value_counts().unstack(fill_value=0)
    f["cancel_rate"] = (status_counts["cancelled"] / status_counts.sum(axis=1)).reindex(f.index)
    f = f.join(c.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
    f["tenure_days"] = (CUTOFF - f.pop("signup_date")).dt.days

    sept = o[(o.status == "completed") & (o.order_ts >= CUTOFF) & (o.order_ts < SEPT_END)]
    f["lapsed"] = (~f.index.isin(sept["customer_id"])).astype(int)   # label only, never a feature
    return f.reset_index()


def lapsed_metrics(y_true, y_pred):
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def show(name, y_true, y_pred):
    m = lapsed_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"{name}: accuracy {accuracy_score(y_true, y_pred):.3f}  lapsed precision {m['precision']:.3f}  "
          f"recall {m['recall']:.3f}  F1 {m['f1']:.3f}")
    print(f"  confusion [rows = actual 0/1, cols = predicted 0/1]: {cm.tolist()}")
    return m


def make_pipeline(model):
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(drop="first"), NOMINAL)])
    return Pipeline([("prep", pre), ("model", model)])


if __name__ == "__main__":
    data = build_dataset()
    X, y = data[NUMERIC + NOMINAL], data["lapsed"]
    print(f"=== cohort ===\ncustomers {len(data)}; lapsed {y.sum()} ({y.mean():.2%})")
    assert X.isna().sum().sum() == 0, "a feature column has a missing value"
    print("no feature column contains a missing value")

    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(X, y, data["customer_id"], test_size=0.2,
                                                            stratify=y, random_state=42)
    print(f"train {len(X_tr)} rows, test {len(X_te)} rows")

    print("\n=== baseline: always 'not lapsed' ===")
    show("baseline", y_te, np.zeros(len(y_te), dtype=int))

    print("\n=== tuning (5-fold CV, scoring = F1 of lapsed class) ===")
    lr = GridSearchCV(make_pipeline(LogisticRegression(max_iter=1000)),
                      {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
    rf = GridSearchCV(make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
                      {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
                      cv=5, scoring="f1").fit(X_tr, y_tr)
    print(f"logistic regression best {lr.best_params_}  CV F1 {lr.best_score_:.3f}")
    print(f"random forest       best {rf.best_params_}  CV F1 {rf.best_score_:.3f}  "
          f"({len(rf.cv_results_['params']) * 5} fits)")

    scaler = lr.best_estimator_.named_steps["prep"].named_transformers_["num"]
    print(f"\nscaler mean of frequency (train rows only): {scaler.mean_[NUMERIC.index('frequency')]:.3f}  "
          f"vs full-cohort mean {X['frequency'].mean():.3f}")

    print("\n=== test set (used once, after tuning) ===")
    lr_m = show("logistic regression", y_te, lr.predict(X_te))
    rf_m = show("random forest", y_te, rf.predict(X_te))
    best, best_name = (rf, "random forest") if rf_m["f1"] >= lr_m["f1"] else (lr, "logistic regression")
    print(f"selected: {best_name}")

    print("\n=== class balance: over-sample lapsed training rows ===")
    rng = np.random.RandomState(42)
    pos = np.where(y_tr.values == 1)[0]
    extra = rng.choice(pos, size=(y_tr == 0).sum() - len(pos), replace=True)
    X_bal = pd.concat([X_tr, X_tr.iloc[extra]])
    y_bal = pd.concat([y_tr, y_tr.iloc[extra]])
    print(f"balanced training counts: {y_bal.value_counts().to_dict()}")
    lr_bal = make_pipeline(LogisticRegression(max_iter=1000, C=lr.best_params_["model__C"])).fit(X_bal, y_bal)
    show("logistic regression, unbalanced", y_te, lr.predict(X_te))
    show("logistic regression, balanced  ", y_te, lr_bal.predict(X_te))

    print("\n=== interpretation ===")
    rf_model = rf.best_estimator_
    names = rf_model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf_model.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
    print("random forest top 5 importances:")
    print(imp.head(5).round(3).to_string())
    coef = pd.Series(lr.best_estimator_.named_steps["model"].coef_[0],
                     index=lr.best_estimator_.named_steps["prep"].get_feature_names_out())
    print("logistic regression coefficients (positive = more likely to lapse):")
    print(coef.round(3).sort_values().to_string())

    joblib.dump(best.best_estimator_, "outputs/lapse_model.joblib")
    reloaded = joblib.load("outputs/lapse_model.joblib")
    same = (reloaded.predict(X_te) == best.predict(X_te)).all()
    print(f"\nsaved outputs/lapse_model.joblib; reloaded model reproduces test predictions exactly: {same}")
    assert same

    seg = pd.read_csv("outputs/customer_segments.csv")
    premium = set(seg.loc[seg.is_premium, "customer_id"])
    test = pd.DataFrame({"customer_id": id_te.values, "lapse_probability": reloaded.predict_proba(X_te)[:, 1].round(4),
                         "actually_lapsed": y_te.values})
    retention = (test[test.customer_id.isin(premium)].sort_values("lapse_probability", ascending=False).head(10))
    retention.to_csv("outputs/retention_list.csv", index=False)
    print(f"\nPremium customers in test set: {test.customer_id.isin(premium).sum()}; top {len(retention)} written to "
          f"outputs/retention_list.csv; truly lapsed among them: {retention.actually_lapsed.sum()}")
    print(retention.to_string(index=False))
