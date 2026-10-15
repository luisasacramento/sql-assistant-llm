"""Junta treino + avaliação em results/summary.csv e gera gráficos em docs/."""
import glob, json, os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for f in glob.glob("results/eval_*.json"):
    e = json.load(open(f, encoding="utf-8"))
    meta_f = f"results/{e['name']}/meta.json"
    if not os.path.exists(meta_f) and e["adapter"]:
        meta_f = os.path.join(os.path.dirname(e["adapter"]), "meta.json")
    m = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
    rows.append({"experimento": e["name"], "rank": m.get("rank"), "lr": m.get("lr"), "epocas": m.get("epochs"),
                 "temperatura": e["temperature"], "loss_final": m.get("final_loss"),
                 "tempo_treino_s": m.get("train_time_s"), "vram_pico_gb": m.get("peak_vram_gb"),
                 "exact_match": e["exact_match"], "sql_valido": e["valid_sql"]})
df = pd.DataFrame(rows).sort_values("experimento")
df.to_csv("results/summary.csv", index=False)
print(df.to_string(index=False))

os.makedirs("docs", exist_ok=True)
ax = df.set_index("experimento")[["exact_match", "sql_valido"]].plot.bar(figsize=(9, 4), rot=30)
ax.set_ylabel("Taxa"); ax.set_title("Baseline vs experimentos (exemplos de teste)")
plt.tight_layout(); plt.savefig("docs/comparacao_experimentos.png", dpi=150); plt.close()

plt.figure(figsize=(8, 4))
for f in sorted(glob.glob("results/*/log_history.json")):
    h = [l for l in json.load(open(f)) if "loss" in l]
    plt.plot([l["step"] for l in h], [l["loss"] for l in h], label=f.split(os.sep)[1])
plt.xlabel("step"); plt.ylabel("loss"); plt.title("Curvas de loss"); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig("docs/curvas_loss.png", dpi=150)
