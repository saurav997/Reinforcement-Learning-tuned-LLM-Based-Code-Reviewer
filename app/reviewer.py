# reviewer.py
import os, time, threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch, jsonlines

MODEL_NAME = os.getenv("GEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_tok, _mdl = None, None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load():
    global _tok, _mdl
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME).to(device)
        _mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    return _tok, _mdl

PROMPT = """You are a senior Python reviewer. Given the code, write a brief review with 1–3 specific, actionable suggestions. Be concise.

CODE:
{code}

REVIEW:
"""

def generate_one_stream(code: str, max_new_tokens=180, temperature=0.8, top_p=0.9):
    """Yield tokens for a single review in real time."""
    tok, mdl = _load()
    prompt = PROMPT.format(code=code)
    inputs = tok(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        streamer=streamer,
    )

    thread = threading.Thread(target=mdl.generate, kwargs=gen_kwargs)
    thread.start()

    # The streamer yields chunks (substrings). We pass them through.
    for piece in streamer:
        yield piece

def generate_candidates(code: str, k=3):
    """Non-streaming multi-candidate generation for ranking."""
    tok, mdl = _load()
    out = []
    for _ in range(k):
        inputs = tok(PROMPT.format(code=code), return_tensors="pt")
        seq = mdl.generate(**inputs, max_new_tokens=180, do_sample=True, temperature=0.8, top_p=0.9)
        txt = tok.decode(seq[0], skip_special_tokens=True)
        out.append(txt.split("REVIEW:")[-1].strip())
    return out

def select_best(scored):
    return max(scored, key=lambda x: x["total"])

def log_event(event: dict, path="logs/events.jsonl"):
    os.makedirs("logs", exist_ok=True)
    event["ts"] = time.time()
    with jsonlines.open(path, "a") as w:
        w.write(event)
