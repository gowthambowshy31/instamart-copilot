"""Part D, tasks 5-8: tools, ReAct refund agent with safety limits, checkpoints and resume.
Run from the repository root (after Part A):  python src/part_d_agent.py"""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_llm  # noqa: E402
from part_d_rag import retrieve  # noqa: E402  (same retrieval as the policy assistant)

DB, LEDGER, RUNS, STATE = "data/instamart.db", "outputs/refund_ledger.jsonl", "outputs/agent_runs.jsonl", "state"
SYSTEM = ("You are the Instamart refund AGENT. Decide refund requests by calling tools: look up the order, "
          "check the refund policy, calculate the refund, then issue it or escalate it to a human. "
          "Finish with one line that starts with REFUND, ESCALATE, DENIED or NOT_FOUND and a colon.")


# ---------------- tools ----------------
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return dict(zip(["order_id", "status", "order_amount", "delivery_minutes"], row))


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Completed order delivered in more than 45 minutes -> 100 % refund, otherwise 0."""
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


def _spec(name, description, **params):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {k: {"type": t} for k, t in params.items()},
                           "required": list(params)}}


TOOLS_SPEC = [
    _spec("get_order", "Look up one order in the database by its order id.", order_id="string"),
    _spec("search_policy", "Return the three policy passages that best match a query.", query="string"),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          order_amount="number", delivery_minutes="number"),
    _spec("issue_refund", "Record an approved refund of INR 1000 or less in the refund ledger.",
          order_id="string", refund_inr="number"),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead for approval.",
          order_id="string", refund_inr="number"),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ---------------- agent loop ----------------
class SimulatedCrash(Exception):
    pass


def _save_checkpoint(s):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, f"{s['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1, default=str)


def _fallback(s, status, reason):
    done = ", ".join(st["tool"] for st in s["steps"]) or "none"
    s["status"] = status
    s["final_answer"] = (f"Handing over to a human agent: {reason} after {len(s['steps'])} tool steps "
                         f"(tools run: {done}). Customer request: {s['user_message']}")
    return s


def _loop(s, crash_after_tool=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= s["time_limit_seconds"]:
            return _fallback(s, "FALLBACK_TIMEOUT", "time limit reached")
        if s["llm_calls"] >= s["max_iterations"]:
            return _fallback(s, "FALLBACK_MAX_ITERATIONS", f"no final answer within {s['max_iterations']} model calls")
        response = mock_llm.chat(s["messages"], tools=TOOLS_SPEC)
        s["llm_calls"] += 1
        s["tokens_used"] += response.usage.total_tokens
        if s["tokens_used"] > s["token_limit"]:
            return _fallback(s, "FALLBACK_TOKEN_LIMIT", f"token limit {s['token_limit']} exceeded")
        msg = response.choices[0].message
        if not msg.tool_calls:
            s["final_answer"] = msg.content
            s["status"] = msg.content.split(":")[0].strip()
            return s
        calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
        s["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        for tc in calls:
            observation = TOOL_MAP[tc["name"]](**tc["arguments"])
            s["messages"].append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"],
                                  "content": json.dumps(observation)})
            s["steps"].append({"tool": tc["name"], "arguments": tc["arguments"], "observation": observation})
            s["tools_executed"].append(tc["name"])
            s["step"] = len(s["steps"])
            _save_checkpoint(s)
            if crash_after_tool is not None and s["step"] == crash_after_tool:
                raise SimulatedCrash(f"simulated crash after tool step {s['step']}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tool=None):
    s = {"run_id": run_id or f"{scenario_id}-{datetime.now():%Y%m%d%H%M%S%f}", "scenario_id": scenario_id,
         "user_message": user_message, "max_iterations": max_iterations, "token_limit": token_limit,
         "time_limit_seconds": time_limit_seconds, "messages": [{"role": "system", "content": SYSTEM},
                                                                {"role": "user", "content": user_message}],
         "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": [],
         "status": None, "final_answer": None}
    return _loop(s, crash_after_tool)


def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        s = json.load(f)
    return _loop(s)


def refund_amount(result):
    return next((st["observation"]["refund_inr"] for st in result["steps"]
                 if st["tool"] in ("issue_refund", "escalate_to_human")), 0.0)


def log_run(result, note=""):
    line = {"scenario_id": result["scenario_id"], "run_id": result["run_id"], "note": note,
            "status": result["status"], "final_answer": result["final_answer"], "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"], "refund_inr": refund_amount(result),
            "steps": result["steps"]}
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, default=str) + "\n")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    for path in (LEDGER, RUNS):                      # fresh files for a reproducible run
        if os.path.exists(path):
            os.remove(path)
    if os.path.isdir(STATE):
        for fn in os.listdir(STATE):
            os.remove(os.path.join(STATE, fn))

    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    rows = []
    print("=== seven scenarios ===")
    for sc in scenarios.itertuples():
        r = run_agent(sc.message, sc.scenario_id, run_id=sc.scenario_id)
        log_run(r, "scenario")
        expected_ok = (r["status"] == sc.expected_outcome or
                       (sc.expected_outcome == "FALLBACK" and r["status"].startswith("FALLBACK")))
        amount_ok = abs(refund_amount(r) - sc.expected_refund_inr) <= 0.01
        rows.append({"scenario": sc.scenario_id, "expected": sc.expected_outcome, "status": r["status"],
                     "refund_inr": refund_amount(r), "model_calls": r["llm_calls"], "tokens": r["tokens_used"],
                     "ok": expected_ok and amount_ok})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    assert table.ok.all(), "a scenario did not give its expected outcome"

    s1 = scenarios.iloc[0]
    print("\n=== stop-safely demos ===")
    r = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token-limit-200")
    log_run(r, "demo token_limit=200")
    print(f"S1 with token_limit=200: {r['status']} after {r['llm_calls']} model calls, {r['tokens_used']} tokens, "
          f"tools run {r['tools_executed']}")
    r = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout-0")
    log_run(r, "demo time_limit_seconds=0")
    print(f"S1 with time_limit_seconds=0: {r['status']} after {r['llm_calls']} model calls")
    print(f"  final_answer: {r['final_answer']}")

    print("\n=== crash and resume (S1) ===")
    try:
        run_agent(s1.message, "S1", run_id="S1-crash-resume", crash_after_tool=2)
    except SimulatedCrash as e:
        before = json.load(open(os.path.join(STATE, "S1-crash-resume.json")))["tools_executed"]
        print(f"crashed: {e}; tools executed before crash: {before}")
    r = resume_agent("S1-crash-resume")
    log_run(r, "demo crash-and-resume")
    print(f"after resume: status {r['status']}, refund INR {refund_amount(r)}, "
          f"tools executed across both parts: {r['tools_executed']}")
    assert r["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert r["status"] == "REFUND" and refund_amount(r) == 128.25

    print(f"\nledger lines: {sum(1 for _ in open(LEDGER))}; agent run lines: {sum(1 for _ in open(RUNS))}; "
          f"checkpoints in state/: {len(os.listdir(STATE))}")
