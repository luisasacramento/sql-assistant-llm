"""Avalia modelo base (sem --adapter) ou fine-tunado no conjunto de teste."""
import argparse, json, os, torch
from common import DEFAULT_MODEL, build_prompt, clean_sql, load_model, load_tokenizer, normalize, valid_sql

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--adapter", default=None)
ap.add_argument("--name", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()

tok = load_tokenizer(a.model)
tok.padding_side = "left"
model = load_model(a.model, a.adapter)
data = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")][: a.n]

preds = []
for i in range(0, len(data), a.bs):
    batch = data[i:i + a.bs]
    enc = tok([build_prompt(e["context"], e["question"]) for e in batch],
              return_tensors="pt", padding=True).to("cuda")
    kw = dict(do_sample=True, temperature=a.temperature, top_p=0.9) if a.temperature > 0 else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=96, pad_token_id=tok.pad_token_id, **kw)
    preds += [clean_sql(t) for t in tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    print(f"{len(preds)}/{len(data)}", end="\r")

em = sum(normalize(p) == normalize(e["answer"]) for p, e in zip(preds, data)) / len(data)
vs = sum(valid_sql(e["context"], p) for p, e in zip(preds, data)) / len(data)
res = {"name": a.name, "adapter": a.adapter, "temperature": a.temperature, "n": len(data),
       "exact_match": round(em, 4), "valid_sql": round(vs, 4),
       "samples": [{"question": e["question"], "context": e["context"], "expected": e["answer"], "pred": p}
                   for e, p in zip(data, preds)]}
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/eval_{a.name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n{a.name}: exact_match={em:.3f} valid_sql={vs:.3f}")
