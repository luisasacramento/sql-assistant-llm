"""Gera data/train.jsonl e data/test.jsonl a partir do sql-create-context."""
import json, argparse
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=5000)
ap.add_argument("--n_test", type=int, default=200)
a = ap.parse_args()

ds = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=42)
test = ds.select(range(a.n_test))
train = ds.select(range(a.n_test, a.n_test + a.n_train))  # sem sobreposição com o teste

for name, d in [("train", train), ("test", test)]:
    with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
        for ex in d:
            f.write(json.dumps({k: ex[k] for k in ("question", "context", "answer")}, ensure_ascii=False) + "\n")
    print(name, len(d))
