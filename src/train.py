"""Fine-tuning QLoRA para Text-to-SQL. Salva adapter, log de loss, tempo e pico de VRAM."""
import argparse, json, os, time, torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments)
from common import DEFAULT_MODEL, bnb_config, build_prompt, load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=float, default=1)
ap.add_argument("--n_train", type=int, default=5000)
ap.add_argument("--bs", type=int, default=2, help="batch por passo (diminua se faltar VRAM)")
ap.add_argument("--eff_bs", type=int, default=16, help="batch efetivo (gradient accumulation)")
ap.add_argument("--max_len", type=int, default=384)
ap.add_argument("--name", default=None)
a = ap.parse_args()

assert torch.cuda.is_available(), "GPU não detectada! Veja docs/SETUP_GPU.md"
name = a.name or f"r{a.rank}_lr{a.lr}_ep{a.epochs}"
out = f"results/{name}"
os.makedirs(out, exist_ok=True)

tok = load_tokenizer(a.model)

def tokenize(ex):
    p = tok(build_prompt(ex["context"], ex["question"]), add_special_tokens=False)["input_ids"]
    r = tok(ex["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + r)[: a.max_len]
    labels = ([-100] * len(p) + r)[: a.max_len]   # loss só na resposta
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

ds = load_dataset("json", data_files="data/train.jsonl")["train"].select(range(a.n_train))
ds = ds.map(tokenize, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb_config(), device_map={"": 0})
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=out, num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
    gradient_accumulation_steps=max(1, a.eff_bs // a.bs), learning_rate=a.lr, fp16=True,
    lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10, save_strategy="no",
    report_to="none", remove_unused_columns=False)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()
elapsed = time.time() - t0

model.save_pretrained(f"{out}/adapter")
losses = [l["loss"] for l in trainer.state.log_history if "loss" in l]
json.dump(trainer.state.log_history, open(f"{out}/log_history.json", "w"))
json.dump({"name": name, "model": a.model, "rank": a.rank, "lr": a.lr, "epochs": a.epochs,
           "eff_bs": a.eff_bs, "n_train": a.n_train, "train_time_s": round(elapsed, 1),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
           "final_loss": round(sum(losses[-5:]) / max(1, len(losses[-5:])), 4)},
          open(f"{out}/meta.json", "w"), indent=2)
print("Treino concluído em", round(elapsed), "s ->", out)
