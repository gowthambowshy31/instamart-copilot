"""Part C: support-ticket triage around mock_llm.py with guardrails, redaction, retries,
output validation, a token budget and segment-aware routing.
Run from the repository root (after Part B):  python src/part_c_triage.py"""
import json
import os
import random
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.getcwd())
import mock_llm  # noqa: E402  (supplied file, used unmodified)

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
HUMAN_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

PII = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
       ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
       ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO_SHOT = """You CLASSIFY customer support tickets for a grocery delivery app.
Allowed categories: billing, delivery, technical, fraud.
Reply with JSON only, exactly in the form {"category": "...", "reason": "..."}."""

FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: Someone used my card for an order I never placed. -> {"category": "fraud", "reason": "unauthorized use"}
Ticket: The app freezes on the payment page. -> {"category": "technical", "reason": "app error"}"""


def redact(text):
    counts = {}
    for name, pattern, placeholder in PII:
        text, counts[name] = pattern.subn(placeholder, text)
    return text, counts


def call_with_retry(messages, ticket_id, jitter):
    """Exponential backoff with jitter: wait before retry n = BASE_DELAY * 2**n * U(0.9, 1.15)."""
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages), n
        except mock_llm.TransientLLMError as e:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * jitter.uniform(0.9, 1.15)
            print(f"  retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s ({e})")
            time.sleep(wait)


def validate(reply):
    """Return (status, category). Never guess: anything unclear goes to a human."""
    try:
        data = json.loads(reply)
        if isinstance(data, dict) and data.get("category") in CATEGORIES:
            return "classified_json", data["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        found = set(re.findall(r"[a-z]+", (reply or "").lower())) & set(CATEGORIES)
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""


def run_pipeline(tickets, system_prompt, token_budget=None, label=""):
    mock_llm.reset()
    jitter = random.Random(42)
    total_tokens, retries, rows = 0, 0, []
    for t in tickets.itertuples():
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": "", "category": "", "tokens_used": 0}
        if not t.text.strip():
            row["status"] = "skipped_empty"
        elif mock_llm.est_tokens(t.text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            clean, _ = redact(t.text)
            assert not any(p.search(clean) for _, p, _ in PII), f"PII left in prompt for {t.ticket_id}"
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            response, n_retries = call_with_retry(messages, t.ticket_id, jitter)
            retries += n_retries
            row["tokens_used"] = response.usage.total_tokens
            total_tokens += response.usage.total_tokens
            row["status"], row["category"] = validate(response.choices[0].message.content)
        rows.append(row)
    out = pd.DataFrame(rows)
    print(f"[{label}] transient errors recovered by retry: {retries}; total tokens {total_tokens}")
    return out


def summarise(df, label):
    done = df[df.status.isin(["classified_json", "classified_keyword"])]
    correct = (done.category == done.true_category).sum()
    print(f"[{label}] status counts: {df.status.value_counts().to_dict()}")
    print(f"[{label}] classified {len(done)}; accuracy {correct}/{len(done)} = {correct / len(done):.1%}; "
          f"tokens {df.tokens_used.sum()}; tokens per classified ticket {df.tokens_used.sum() / len(done):.1f}")
    return done


def route(df):
    seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
    df = df.copy()
    df["segment_name"] = df.customer_id.map(seg["segment_name"]).fillna("")
    premium = df.customer_id.map(seg["is_premium"]).fillna(False).astype(bool)
    human = df.status.isin(HUMAN_STATUSES)
    df["route"] = "support_queue"
    df.loc[premium, "route"] = "premium_desk"
    df.loc[df.category == "fraud", "route"] = "security_team"
    df.loc[human, "route"] = "human_review"
    df["priority"] = df.route.map({"security_team": "HIGH", "premium_desk": "HIGH"}).fillna("NORMAL")
    return df[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]


if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    os.makedirs("outputs", exist_ok=True)

    print("=== input guardrails ===")
    empty = tickets.text.str.strip() == ""
    long_ = tickets.text.map(mock_llm.est_tokens) > MAX_INPUT_TOKENS
    print(f"empty tickets: {empty.sum()}; long tickets: {long_.sum()} "
          f"(estimated tokens {sorted(set(tickets[long_].text.map(mock_llm.est_tokens)))})")

    print("\n=== redaction over all 164 tickets ===")
    totals = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
    redacted = []
    for text in tickets.text:
        clean, counts = redact(text)
        redacted.append(clean)
        for k in totals:
            totals[k] += counts[k]
    left = sum(len(p.findall(t)) for t in redacted for _, p, _ in PII)
    print(f"substitutions {totals}, total {sum(totals.values())}; matches left after redaction: {left}")
    assert left == 0

    print(f"\nzero-shot prompt: {mock_llm.est_tokens(ZERO_SHOT)} est. tokens; "
          f"few-shot prompt: {mock_llm.est_tokens(FEW_SHOT)} est. tokens")

    print("\n=== zero-shot full run (no budget) ===")
    zero = run_pipeline(tickets, ZERO_SHOT, label="zero-shot")
    zero_done = summarise(zero, "zero-shot")
    print("zero-shot confusion (rows = true, cols = predicted):")
    print(pd.crosstab(zero_done.true_category, zero_done.category).to_string())

    print("\n=== few-shot full run (no budget) ===")
    few = run_pipeline(tickets, FEW_SHOT, label="few-shot")
    summarise(few, "few-shot")

    print("\n=== budget demo: zero-shot, TOKEN_BUDGET = 6000 ===")
    demo = run_pipeline(tickets, ZERO_SHOT, token_budget=6000, label="budget-6000")
    print(f"budget_exhausted tickets: {(demo.status == 'budget_exhausted').sum()}")
    route(demo).to_csv("outputs/triage_budget_demo.csv", index=False)

    print("\n=== final: few-shot, TOKEN_BUDGET = 100000, with routing ===")
    final = route(run_pipeline(tickets, FEW_SHOT, token_budget=100000, label="final"))
    final.to_csv("outputs/triage_results.csv", index=False)
    print(f"wrote outputs/triage_results.csv ({len(final)} rows)")
    print(final.groupby(["route", "priority"]).size().to_string())
