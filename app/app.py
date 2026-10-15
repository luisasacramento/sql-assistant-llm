"""Frontend Gradio: python app/app.py  -> http://127.0.0.1:7860"""
import glob, os, sys, torch
import gradio as gr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import DEFAULT_MODEL, build_prompt, clean_sql, load_model, load_tokenizer

adapters = sorted(glob.glob("results/*/adapter"))
if not adapters:
    sys.exit("Nenhum adapter encontrado em results/. Rode src/train.py primeiro.")
tok = load_tokenizer(DEFAULT_MODEL)
tok.padding_side = "left"
model = load_model(DEFAULT_MODEL, adapters[0])      # adapter_name = caminho do adapter
for ad in adapters[1:]:
    model.load_adapter(ad, adapter_name=ad)

def gerar(schema, pergunta, adapter, usar_base, temperatura, max_tokens):
    enc = tok(build_prompt(schema, pergunta), return_tensors="pt").to("cuda")
    kw = dict(do_sample=True, temperature=temperatura, top_p=0.9) if temperatura > 0 else dict(do_sample=False)
    def _gen():
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=int(max_tokens), pad_token_id=tok.pad_token_id, **kw)
        return clean_sql(tok.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=True))
    if usar_base:
        with model.disable_adapter():
            return _gen()
    model.set_adapter(adapter)
    return _gen()

ex = [["CREATE TABLE alunos (id INTEGER, nome VARCHAR, nota INTEGER)", "Quantos alunos têm nota maior que 7?"],
      ["CREATE TABLE pedidos (id INTEGER, cliente VARCHAR, valor FLOAT)", "Qual a soma dos valores dos pedidos do cliente Ana?"]]

gr.Interface(
    fn=gerar, title="Text-to-SQL com LLM local (Qwen2.5 + QLoRA)", examples=ex,
    inputs=[gr.Textbox(label="Schema (CREATE TABLE ...)", lines=4), gr.Textbox(label="Pergunta"),
            gr.Dropdown(adapters, value=adapters[0], label="Adapter (experimento)"),
            gr.Checkbox(label="Usar modelo BASE (sem fine-tuning)"),
            gr.Slider(0, 1, 0, step=0.1, label="Temperatura"), gr.Slider(32, 256, 96, step=16, label="Máx. tokens")],
    outputs=gr.Code(label="SQL gerado", language="sql"),
).launch(share=os.environ.get("GRADIO_SHARE") == "1")
