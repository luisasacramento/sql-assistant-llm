import re, sqlite3, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def build_prompt(context, question):
    return f"### Schema:\n{context}\n### Pergunta:\n{question}\n### SQL:\n"

def clean_sql(text):
    text = text.split("###")[0]
    if ";" in text:
        text = text.split(";")[0]
    return text.strip()

def normalize(sql):
    s = sql.lower().strip().rstrip(";")
    s = s.replace('"', "'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([(),=<>])\s*", r"\1", s)
    return s

def valid_sql(context, sql):
    """True se a query é sintaticamente válida para o schema (SQLite EXPLAIN)."""
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(context if context.strip().endswith(";") else context + ";")
        con.execute("EXPLAIN " + sql)
        return True
    except Exception:
        return False

def bnb_config():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)

def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_model(model_name, adapter=None):
    assert torch.cuda.is_available(), "GPU não detectada! Veja docs/SETUP_GPU.md"
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config(),
                                                 device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, adapter_name=adapter)
    model.eval()
    return model
