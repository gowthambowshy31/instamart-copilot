"""Part D, tasks 1-4: chunk the policy documents, TF-IDF retrieval, grounded answers, evaluation.
Run from the repository root:  python src/part_d_rag.py
The refund agent (part_d_agent.py) imports retrieve() from here."""
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.getcwd())
import mock_llm  # noqa: E402

POLICY_DIR = "data/policies"
SYSTEM = ("You are the Instamart POLICY ASSISTANT. Answer only from the supplied context. "
          "If the context does not contain the answer, say: I could not find that information in the policy documents.")


def chunk_words(words, size=40, overlap=10):
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):          # this chunk reaches the last word
            return chunks
        start += size - overlap


class PolicyIndex:
    def __init__(self, size=40, overlap=10):
        self.chunks = []
        for fn in sorted(os.listdir(POLICY_DIR)):
            words = open(os.path.join(POLICY_DIR, fn), encoding="utf-8").read().split()
            self.chunks += [{"source": fn, "text": t} for t in chunk_words(words, size, overlap)]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 4),
                 "text": self.chunks[i]["text"]} for i in best]

    def answer(self, question):
        hits = self.retrieve(question, k=3)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
        reply = mock_llm.chat(messages).choices[0].message.content
        return reply, [h["source"] for h in hits]


DEFAULT = None


def retrieve(query, k=3):
    global DEFAULT
    if DEFAULT is None:
        DEFAULT = PolicyIndex()
    return DEFAULT.retrieve(query, k)


def evaluate(index, label):
    qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    rows = []
    for q in qa.itertuples():
        reply, sources = index.answer(q.question)
        answerable = q.expected_doc != "NONE"
        rows.append({"question": q.question, "answerable": answerable,
                     "doc_in_top3": answerable and q.expected_doc in sources,
                     "fact_in_answer": answerable and q.expected_fact in reply,
                     "fallback": reply.startswith("I could not find"), "answer": reply})
    r = pd.DataFrame(rows)
    a, u = r[r.answerable], r[~r.answerable]
    print(f"[{label}] expected doc in top 3: {a.doc_in_top3.sum()}/{len(a)}; "
          f"fact in answer: {a.fact_in_answer.sum()}/{len(a)}; "
          f"unanswerable -> fallback: {u.fallback.sum()}/{len(u)}")
    for row in a[~(a.doc_in_top3 & a.fact_in_answer)].itertuples():
        print(f"  FAILED: {row.question}\n    answer: {row.answer}")
    return r


if __name__ == "__main__":
    index = PolicyIndex()
    counts = pd.Series([c["source"] for c in index.chunks]).value_counts()
    print(f"=== chunks (40 words, overlap 10) ===\ntotal {len(index.chunks)}: {counts.to_dict()}")

    print("\n=== retrieve('refund policy late delivery') ===")
    for h in index.retrieve("refund policy late delivery"):
        print(f"  {h['source']}  score {h['score']}")

    print("\n=== evaluation ===")
    evaluate(index, "overlap 10")
    zero = PolicyIndex(overlap=0)
    print(f"zero-overlap index: {len(zero.chunks)} chunks")
    evaluate(zero, "overlap 0")
