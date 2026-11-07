# Instamart Customer Intelligence & Support Copilot

## Overview

One internal tool for Instamart's Customer Experience team, built in four connected Parts. All data is synthetic, and all money is in Indian Rupees (INR).

| Part | What it does | Main outputs |
|---|---|---|
| A. Data foundation | Audits and cleans the raw exports, stores them in SQLite, answers 7 business questions in SQL and pandas | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B. Segments and lapse prediction | Groups customers with RFM + K-means; predicts who will stop ordering | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C. Ticket triage | Sorts support tickets with guardrails, PII redaction, retries, output checks, a token budget and segment-aware routing | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D. Policy assistant and refund agent | TF-IDF retrieval over 5 policies with grounded answers; a ReAct refund agent with tools, safety limits and crash-resume | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

Each Part feeds the next: B reads A's clean CSVs, C routes using B's segments, and D's agent reads A's database. Every model call goes through the supplied offline `mock_llm.py`, so no API key, network or money is needed. `generate_data.py` and `mock_llm.py` are copied unmodified from the brief.

```
generate_data.py, mock_llm.py   supplied files (unmodified)
src/part_a_clean.py             A: audit, clean, store
src/validate.py                 A: checkpoint assertions -> ALL CHECKPOINTS PASSED
src/part_a_analysis.py          A: runs sql/queries.sql, pandas cross-check, statistics
src/part_b_segments.py          B: RFM, K-means, PCA, segment file
src/part_b_lapse.py             B: lapse features, tuned models, balancing, saved model, retention list
src/part_c_triage.py            C: triage pipeline, zero/few-shot comparison, budget demo, routing
src/part_d_rag.py               D: chunking, TF-IDF retrieval, grounded answers, evaluation
src/part_d_agent.py             D: tools, ReAct agent, stop-safely demos, crash and resume
```

## Reproduce

Needs Python 3.11 or newer. Run from the repository root. All seeds are fixed, so the numbers below repeat exactly.

1. `python -m venv .venv`
2. `source .venv/bin/activate` (on Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python src/part_a_clean.py`
6. `python src/validate.py`
7. `python src/part_a_analysis.py`
8. `python src/part_b_segments.py`
9. `python src/part_b_lapse.py`
10. `python src/part_c_triage.py`
11. `python src/part_d_rag.py`
12. `python src/part_d_agent.py`

Step 4 ends with `generated: customers=1200 products=40 orders_raw=13084 tickets=164`, and step 6 prints `ALL CHECKPOINTS PASSED`. The whole run takes under a minute.

## Part A

### Raw data audit

Shape and missing values of the three raw tables (`orders.csv` is read so that only truly blank cells count as missing, so the text `N/A` stays visible as a bad amount):

| Table | Shape | Column (dtype): missing count (%) |
|---|---|---|
| customers.csv | 1,200 × 5 | customer_id (str): 0 · city (str): 0 · signup_date (str): 0 · age (float): 61 (5.08%) · gender (str): 30 (2.50%) |
| products.csv | 40 × 4 | product_id (str), product_name (str), category (str), price_inr (int): 0 missing in every column |
| orders.csv | 13,084 × 9 | order_id, customer_id, product_id (str): 0 · quantity (int): 0 · order_ts (str): 0 · status (str): 0 · discount_pct (int): 0 · order_amount (str): 0 · delivery_minutes (float): 2,541 (19.42%) |

| Audit figure | Value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status `pending` | 656 |
| distinct order_id | 12,309 |
| order_amount that cannot become a number (`N/A`, `error`) | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10–100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank raw gender spellings | 10 |
| blank gender | 30 |

### Cleaning rules (in the order applied)

**Customers**

| Rule | Values changed |
|---|---|
| City: strip spaces, then title case → exactly 6 cities | 476 |
| Gender: `male/Male/M/m/MALE` → male, `female/Female/F/f/FEMALE` → female, blank → unknown | 950 |
| Age outside 10–100 → missing | 6 |
| Missing age → median age of the customer's city (`groupby("city").transform("median")`) | 67 (61 blank + 6 impossible) |
| signup_date → datetime | 1,200 |

Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; 0 missing ages, all between 18 and 60.

**Orders**

| Step | Rule | Rows or values changed |
|---|---|---|
| 1 | order_ts → datetime | 13,084 values |
| 2 | Remove exact duplicate rows | 119 rows removed → 12,965 |
| 3 | Keep only the latest order_ts per order_id (drops the earlier `pending` copy) | 656 rows removed → 12,309 |
| 4 | order_amount → numeric; unparseable → missing; refill as round(price_inr × quantity × (1 − discount_pct/100), 2) | 112 values rebuilt (the other 8 of the 120 were on removed duplicate rows) |
| 5a | Negative delivery_minutes → missing | 105 values (the 106th was on a removed duplicate row) |
| 5b | Completed orders: missing delivery_minutes → median of the product category among completed orders (Dairy, Electronics, Grocery, Snacks 25.0; Personal Care 25.5) | 774 values |
| 5c | Cancelled orders: delivery_minutes left missing; a cancelled order was never delivered, so a time would be invented data | 1,199 left missing |

Result: 12,309 rows (one per order_id); 11,110 completed and 1,199 cancelled; 0 pending; 0 missing order_amount; 0 missing delivery_minutes among completed orders; total completed order_amount **INR 2,998,668.80**.

### Store and validate

The clean tables are written to `data/clean/` and to `data/instamart.db` as tables `customers`, `products` and `orders`, with order_ts stored as text `YYYY-MM-DD HH:MM:SS`. `src/validate.py` asserts 13 checkpoints: city counts, gender counts, no missing ages, the 18–60 age range, 12,309 unique orders, 11,110 / 1,199 statuses, no pending rows, no missing amounts, no missing completed delivery times, 1,199 missing cancelled delivery times, no negative times, the INR 2,998,668.80 total, and the three database row counts. It prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE then HAVING | 30 customers have more than INR 8,000 in completed orders above INR 500 (top: C0693, INR 13,776.55) |
| Q2 | Three-table join | 30 city × category rows summing to INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 | Scalar subquery | 11 products priced above the average catalogue price |
| Q4 | LEFT JOIN | 25 customers with no completed order |
| Q5 | Two CTEs | 550 customers above the average customer spend of INR 2,552.06 |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER | 8 rows where RANK() ≠ ROW_NUMBER() (top customers below) |
| Q7 | Running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Q6 top customer per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10).

Why 8 rows differ: when customers in the same city have exactly the same total spend, RANK() gives them all the same rank while ROW_NUMBER() still counts 1, 2, 3…, so every tied customer after the first gets a ROW_NUMBER larger than its RANK.

### pandas cross-check and exploration

The `pd.pivot_table(..., margins=True)` grand total is INR 2,998,668.80, equal to the sum of SQL Q2.

| City | Sum (INR) | Mean (INR) | Completed orders |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

| Completed order_amount | Value |
|---|---|
| Mean / median | INR 269.91 / INR 135.00 |
| Skewness | 4.61 |
| Q1 / Q3 / IQR | INR 64.80 / INR 297.00 / INR 232.20 |
| Lower fence / outliers below | INR −283.50 / 0 |
| Upper fence / outliers above | INR 645.30 / 897 |

Correlations over completed orders: order_amount with discount_pct −0.095 (largest in absolute value), order_amount with quantity +0.087, every other pair within ±0.013.

**What is this data about?** Six months (April to September 2025) of quick-commerce orders: 11,110 completed orders from 1,175 customers in six cities, across groceries, dairy, snacks, personal care and electronics.

**What do the distributions look like?** Order value is strongly right-skewed: the mean (INR 269.91) is double the median (INR 135.00), with skewness 4.61. Most baskets are small (half are under INR 135). Delivery times centre on about 25 minutes.

**Are there outliers or anomalies?** 897 orders lie above the INR 645.30 upper fence and none below. They are real electronics purchases (mean INR 1,189.51 per electronics order), not errors. The real anomalies were the impossible values we cleaned: negative delivery times, `N/A` amounts and ages like 999.

**What relationships exist?** All correlations are weak. The strongest is order_amount with discount_pct (−0.095): bigger discounts go with slightly smaller bills. A correlation this small does not show that one causes the other.

## Part B

### RFM table (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency: median 19 days, minimum 0, maximum 180. Frequency: mean 9.46, maximum 33. Monetary: mean INR 2,552.06, maximum INR 14,766.55.

### K-means on standardised RFM

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | **743.1** | **0.526** |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** The elbow is at 4: inertia falls by 525 from K=3 to K=4, but only by 126 from K=4 to K=5. K=4 also has the best silhouette (0.526), and four groups map cleanly onto four campaigns.

Inertia is the total squared distance from each customer to its own centre. Adding centres can only shorten those distances, so K=8 always has lower inertia than K=2, and K=n would reach 0. Inertia alone would therefore always pick the largest K, so we read the elbow, the silhouette (which also penalises clusters that sit too close together) and what the business can act on.

### Segment profile (K = 4)

Cluster numbers are only labels; segment 1 is not "better" or "first" in any way.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Nudge a second weekly order with a small basket-size discount |
| 1 | **Premium Big Spenders** (is_premium) | 138 | 24.12 | 6.77 | 7,129.70 | Priority support line and early access to new electronics |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Monthly grocery subscription with free delivery |
| 3 | Drifting Away | 182 | 109.04 | 2.20 | 685.99 | Time-limited win-back coupon on their favourite category |

PCA(2) on the scaled RFM matrix: explained_variance_ratio_ = **0.634** and **0.218** (total 0.852), so two dimensions keep 85% of the information.

### Lapse problem

- Cutoff: 2025-09-01. Cohort: customers with at least one completed order before the cutoff, which is **1,156** customers.
- Label: `lapsed = 1` if the customer has no completed order from 2025-09-01 to 2025-09-30. That gives **443** lapsed customers (38.32%).
- Features, built only from orders before the cutoff plus the customer table:
  - recency, frequency, monetary
  - average discount_pct, cancellation rate, average delivery_minutes
  - most frequent product category
  - city, age, gender, tenure_days
- Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 924 training rows and 232 test rows.
- Baseline (always "not lapsed"): accuracy 61.6%, lapsed-class F1 0.

### Leakage-free preprocessing

- A `ColumnTransformer` inside a scikit-learn `Pipeline` does the preprocessing:
  - `StandardScaler` on the 8 numeric features.
  - `OneHotEncoder(drop="first")` on city, gender and favourite category, which keeps k−1 columns per variable.
- The pipeline is fitted on training rows only. Inside `GridSearchCV` it is refitted on each training fold.
- **No encoder, scaler or resampler ever saw a test row. No order dated 2025-09-01 or later entered any feature. The segment labels from Part B task 3 are not used as features.**
- The code asserts that no feature column has a missing value.

The fitted scaler's mean of frequency is **8.115**, but the full-cohort mean is **8.029**. They differ because the scaler learned only from the 924 training customers, not from the whole cohort.

### Tuning and test results

| Model | Best parameters | Best CV F1 |
|---|---|---|
| LogisticRegression(max_iter=1000) | C = 100 | 0.651 |
| RandomForestClassifier(n_estimators=100, random_state=42), 45 fits | max_depth = 8, min_samples_leaf = 10 | 0.681 |

The logistic-regression grid is almost flat: C = 0.1, 1, 10 and 100 score 0.646, 0.645, 0.649 and 0.651. That is why our best C (100) differs from the reference (C = 1) without changing the result.

Lapsed class on the untouched test set (232 customers, 89 lapsed):

| Model | Confusion [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | [[143, 0], [89, 0]] | 0.000 | 0.000 | 0.000 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| **Random forest (selected)** | [[113, 30], [31, 58]] | 0.659 | 0.652 | **0.655** |

**Choice: random forest.**
- Both models catch the same 58 of 89 lapsing customers, but the forest raises two fewer false alarms, so it has the higher precision and F1 (0.655, above the 0.55 target).
- A missed lapsing customer loses their whole future basket: the average customer spends INR 2,552 over six months.
- Wrongly targeting a loyal customer only costs one coupon. So recall matters more than precision.
- With equal recall, the forest is the cheaper model to act on. The forest also captures the interaction between "spends a lot" and "orders rarely" without hand-made features.

### Class-balance experiment

- Lapsed training rows were randomly duplicated, with replacement and `random_state=42`, until both classes had 570 rows. Training rows only.
- The best logistic regression (C = 100) was then refitted.

| Logistic regression | Precision | Recall | F1 | Confusion |
|---|---|---|---|---|
| Unbalanced | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Balanced (570 / 570) | 0.589 | **0.854** | 0.697 | [[90, 53], [13, 76]] |

Recall moved most, from 0.65 to 0.85, while precision fell slightly to 0.59. For the retention team, this means catching 76 of the 89 customers who are about to leave instead of 58, at the cost of 21 more coupons sent to customers who would have stayed. If coupons are cheap, the balanced model is the better campaign list.

### Interpretation, saving and retention list

Random forest, five most important features:

| Feature | Importance | Business reading |
|---|---|---|
| frequency | 0.280 | How often someone orders is the strongest sign they will keep ordering |
| monetary | 0.271 | Spend level separates big-basket electronics buyers from daily grocery buyers |
| recency | 0.115 | A long gap since the last order is an early warning |
| avg_delivery | 0.061 | Slower average deliveries are linked to customers drifting away |
| cancel_rate | 0.052 | Customers with many cancellations behave differently from steady buyers |

Logistic regression coefficient signs (on standardised features; positive means more likely to lapse):

| Feature | Coefficient | Business reading |
|---|---|---|
| frequency | −1.698 | Each extra regular order makes lapsing much less likely |
| monetary | +0.285 | At the same frequency, higher spend means a few big one-off purchases, which is a weaker habit |
| avg_delivery | +0.079 | Slower deliveries push customers towards leaving, so delivery speed is a retention lever |

`outputs/lapse_model.joblib` holds the full random-forest pipeline. After `joblib.load`, it reproduces all 232 test predictions exactly (this is asserted in the code).

**Retention list.** The test set has 40 Premium customers. `outputs/retention_list.csv` holds the 10 with the highest predicted lapse probability (0.452 to 0.491). Only **1** of them (C0525) truly lapsed: Premium customers rarely lapse in this data, and none of them crosses the 0.5 threshold. The Premium flag comes from the whole-window RFM, which includes the September window that defines the label. **This retention list is therefore illustrative, not a leakage-free evaluation.**

## Part C

### Guardrails and redaction

- **Input guardrails:**
  - 4 empty tickets get the status `skipped_empty`.
  - 4 long tickets (734 estimated tokens each, over the limit of 400) get `flagged_long`.
  - None of the 8 is sent to the model.
- **Redaction** runs over all 164 tickets, in this order:
  - emails → `[EMAIL]`: 55
  - phone numbers → `[PHONE]`: 91
  - order ids → `[ORDER_ID]`: 132
  - **Total: 278.** Re-applying the three patterns finds 0 matches.
- The user message is exactly the redacted text, and an assertion before every call checks that no PII is left in it.
- **Prompts:**
  - Both contain `CLASSIFY`, list the four categories and require JSON only.
  - Zero-shot: 37 estimated tokens.
  - Few-shot: 78 estimated tokens. It adds an `EXAMPLES:` line with a fraud example and a technical example.

### Resilience and validation

- **Retries:**
  - Every call is wrapped in a retry with at most 3 retries.
  - The wait before retry n is 0.05 × 2ⁿ × uniform(0.9, 1.15) seconds.
  - Each retry prints one log line with the ticket id, retry number and wait.
  - `mock_llm.reset()` runs before each full run.
- **Retry result:** each full run hits exactly 19 `TransientLLMError`s, all recovered on the first retry. All waits were between 0.0453 and 0.0562 s, inside the allowed range of 0.045–0.0575 s. No ticket ends with a permanent error.
- **Validation:** for each of the 156 eligible tickets, a reply is handled in one of three ways:
  - **135 `classified_json`:** JSON with an allowed category.
  - **12 `classified_keyword`:** a chatty sentence in which `re.findall(r"[a-z]+")` finds exactly one category word.
  - **9 `flagged_invalid_output`:** JSON with the made-up category `shipping`, sent to human review. The pipeline never guesses.

### Zero-shot vs few-shot (no budget)

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126 / 147) | 10,180 | 69.3 |
| Few-shot | 147 | **100%** (147 / 147) | 16,576 | 112.8 |

Zero-shot confusion (rows = true category, columns = predicted):

| | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | **11** | 0 | 23 | 0 |
| technical | **10** | 0 | 0 | 29 |

The few-shot prompt costs 6,396 more tokens (+63%, or 43.5 more per ticket) and fixes all 21 errors. That is about 305 extra tokens per corrected ticket.

The extra cost is worth paying here: the 11 fraud tickets that zero-shot calls "billing" would miss the 2-hour security-team rule, and that risk costs far more than a few hundred tokens. For a low-risk queue where every category goes to the same team, zero-shot would be the cheaper choice.

### Budget demo and final routing

- **Budget demo:** with `TOKEN_BUDGET = 6000` (zero-shot), the run stops calling the model once the running total reaches 6,010 tokens. **64 tickets end `budget_exhausted`** and go to human review. See `outputs/triage_budget_demo.csv`.
- **Final run:** `outputs/triage_results.csv` comes from a few-shot run with `TOKEN_BUDGET = 100000`. It has 164 rows. Routing reads `outputs/customer_segments.csv`, and a customer missing from that file counts as not premium.

| Route | Priority | Tickets | Rule |
|---|---|---|---|
| human_review | NORMAL | 17 | skipped_empty (4), flagged_long (4), flagged_invalid_output (9), budget_exhausted (0) |
| security_team | HIGH | 34 | category fraud |
| premium_desk | HIGH | 11 | customer is_premium |
| support_queue | NORMAL | 102 | everything else |

**What was checked by rules and what went to humans:**
- 147 categories were accepted automatically, but only after rule checks: 135 passed JSON parsing and the allowed-category check, and 12 passed the one-category-word check.
- 17 tickets went to humans because no rule could validate them: empty, too long, or an invalid model output.

## Part D

### Retrieval

- **Chunks:** 40 words with an overlap of 10, which gives **15** chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2.
- **Retrieval:** `TfidfVectorizer()` is fitted on the chunks, and `cosine_similarity` ranks them. `retrieve(query, k=3)` returns a source, score and text for each chunk.
- **Check:** for "refund policy late delivery", the top three chunks are all from `refund_policy.txt` (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word appears in a chunk. IDF (inverse document frequency) measures how rare the word is across all chunks, so words that appear everywhere, like "the", count for little. One limitation: TF-IDF matches exact words only, so "money back" does not match a chunk that says "refund", even though they mean the same thing.

`answer(question)` sends two messages:
- A `POLICY ASSISTANT` system message: answer only from the context, otherwise say you could not find it.
- A user message in the form `CONTEXT:` / `[source]` + chunk ×3 / `QUESTION:` / the question.

It returns the reply and the sources used.

### Evaluation (`policy_eval.csv`, 14 questions)

| Setting | Expected doc in top 3 (12 answerable) | Expected fact in answer (12 answerable) | Fallback on the 2 unanswerable |
|---|---|---|---|
| 40 words, overlap 10 | 12 / 12 | 12 / 12 | 2 / 2 |
| 40 words, overlap 0 (13 chunks) | 12 / 12 | **10 / 12** | 2 / 2 |

With zero overlap, two questions break even though the right document is retrieved:
- **"What delivery time makes a completed order eligible for a full refund?"** The sentence "If a completed order is delivered in more than 45 minutes, the customer receives a full refund…" is cut at a chunk boundary. The model only sees "receives a full refund of the order amount." and loses "more than 45 minutes".
- **"How soon is money returned when a payment fails but is deducted?"** "…the amount is returned automatically within 48 hours" is cut before "within 48 hours".

With an overlap of 10, each boundary sentence also appears whole in the next chunk.

### Tools and agent

- `TOOLS_SPEC` lists five tools, each with a name, a one-sentence description and JSON parameters. `TOOL_MAP` maps each name to its Python function.
  - `get_order` reads `data/instamart.db`.
  - `search_policy` reuses the same `retrieve()`.
  - `calculate_refund` applies the refund rule: completed and delivered in more than 45 minutes → 100% refund, otherwise 0.
  - `issue_refund` and `escalate_to_human` each append a timestamped line to `outputs/refund_ledger.jsonl`.
- `run_agent(...)` is a ReAct loop: call the model, run any requested tool, append the observation, and repeat.
- Each run is written as one line to `outputs/agent_runs.jsonl`, with every step's tool, arguments and observation.

| Scenario | Expected | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|---|
| S1 | REFUND | REFUND | 128.25 | 5 | 1,142 |
| S2 | REFUND | REFUND | 135.00 | 5 | 1,142 |
| S3 | ESCALATE | ESCALATE | 1,999.00 | 5 | 1,129 |
| S4 | DENIED | DENIED | 0.00 | 4 | 801 |
| S5 | DENIED | DENIED | 0.00 | 3 | 484 |
| S6 | NOT_FOUND | NOT_FOUND | 0.00 | 2 | 170 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,294 |

### Stop safely

| Demonstration | Status |
|---|---|
| S7 with defaults (max_iterations=6, token_limit=6000) | **FALLBACK_MAX_ITERATIONS**: 6 model calls, 3,294 tokens |
| S1 with token_limit=200 | **FALLBACK_TOKEN_LIMIT**: stopped after the 3rd model call (501 tokens) without running the requested `calculate_refund` |
| S1 with time_limit_seconds=0 | **FALLBACK_TIMEOUT**: stopped before the first model call |

Every fallback returns the steps completed so far and a hand-off summary for a human agent, for example: "Handing over to a human agent: time limit reached after 0 tool steps (tools run: none). Customer request: …".

### Crash and resume (S1)

- After every tool observation, the agent saves a checkpoint to `state/<run_id>.json`. It holds the run id, scenario id, full message list, tokens used, step number and the tools already executed.
- A simulated crash is raised right after the 2nd tool's checkpoint is saved. Then `resume_agent("S1-crash-resume")` loads the checkpoint and continues.

Tools executed, in order:
1. `get_order` (before the crash)
2. `search_policy` (before the crash)
3. `calculate_refund` (after the resume)
4. `issue_refund` (after the resume)

That is four executions with none repeated. The final status is **REFUND, INR 128.25**.

### ReAct vs plan-and-execute

- **Calls and tokens.** Our ReAct agent makes one model call per step: 5 calls for a refund (S1), 3 for a cancelled order (S5) and 2 for an unknown order (S6). Each call resends the growing history, so tokens grow with the number of steps (S1 used 1,142). A plan-and-execute agent asks for the whole plan once (get_order → search_policy → calculate_refund → issue or escalate), then runs the tools without the model. A normal refund would take about 2 calls (plan, then final answer) and fewer tokens.
- **Surprises.** The plan is written before any data is seen. When `get_order` returns NOT_FOUND, or the order turns out to be cancelled, the rest of the plan is wrong. The planner must notice the error and replan, which costs an extra call and needs extra error-handling code. ReAct looks at every observation before choosing the next step, so it stops right away: S6 ends after 2 calls and S5 after 3, with no wasted tool runs.
- **Choice.** For this refund workflow I would keep ReAct. The steps are few, cheap and branch on data, and an unexpected result is common (missing orders, cancellations, refunds over INR 1,000), so adapting after every observation is worth a few hundred extra tokens. The max-iterations, token and time limits already cap ReAct's worst case. Plan-and-execute would pay off for long, predictable workflows where the steps rarely depend on what earlier tools return.

## Limitations and Next Steps

- **The Premium flag is a post-hoc, whole-window label.** It is computed from RFM over April to September, which includes the September window that defines the lapse label. The retention list in `outputs/retention_list.csv` is therefore illustrative, not a leakage-free evaluation. Next step: compute segments from pre-cutoff orders only.
- **The data is synthetic**, built from three fixed customer types, so the segments and lapse patterns are cleaner than real behaviour. Retrain on real exports before trusting the thresholds.
- **The mock model classifies by keywords**, so the 100% few-shot accuracy measures the pipeline, not real language understanding. With a hosted model, re-run the zero-shot vs few-shot comparison and keep the token budget.
- **The lapse model's probabilities** for Premium customers never pass 0.5. A campaign should rank by probability, or use the balanced model (recall 0.85), rather than rely on the default threshold.
- **TF-IDF only matches exact words.** Synonyms ("money back" vs "refund") can miss. Next step: add stemming or embeddings.
- **The refund ledger is not idempotent.** A crash after `issue_refund` runs but before its checkpoint is saved would repeat the refund on resume. Next step: give each refund a unique key and skip it if it is already in the ledger.
- **Refund requests are assumed to be inside the 7-day window**, as the brief states. A real agent must check the request date.
