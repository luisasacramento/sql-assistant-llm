"""Roda baseline + grade de experimentos (treino + avaliação).
Variáveis de ambiente opcionais: N_TRAIN (padrão 5000), MODEL (padrão Qwen2.5-1.5B-Instruct)."""
import os, subprocess, sys

py = sys.executable
N_TRAIN = os.environ.get("N_TRAIN", "5000")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
def run(*c): subprocess.run([py, *c], check=True)

GRID = [  # (nome, rank, lr, épocas)
    ("E1_r8",        8,  2e-4, 1),
    ("E2_r16",       16, 2e-4, 1),
    ("E3_r32",       32, 2e-4, 1),
    ("E4_r16_lr1e4", 16, 1e-4, 1),
    ("E5_r16_ep3",   16, 2e-4, 3),
]

run("src/evaluate.py", "--name", "baseline", "--model", MODEL)
for name, r, lr, ep in GRID:
    run("src/train.py", "--name", name, "--rank", str(r), "--lr", str(lr), "--epochs", str(ep),
        "--n_train", N_TRAIN, "--model", MODEL)
    run("src/evaluate.py", "--name", name, "--adapter", f"results/{name}/adapter", "--model", MODEL)
run("src/evaluate.py", "--name", "E2_r16_temp0.7", "--adapter", "results/E2_r16/adapter",
    "--temperature", "0.7", "--model", MODEL)
run("src/summary.py")
