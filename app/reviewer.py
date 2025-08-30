import os
import time
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import jsonlines

MODEL_NAME = os.getenv("GEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_tok, _mdl = None, None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_lock = threading.Lock()

def _load():
    """Load tokenizer and model, ensuring thread-safe initialization."""
    global _tok, _mdl
    with _lock:
        if _tok is None:
            try:
                _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                _mdl = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=torch_dtype
                ).to(device)
            except Exception as e:
                raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")
    return _tok, _mdl

PROMPT = """You are a senior Python reviewer. Given the code, write a brief review with 1–3 specific, actionable suggestions. Be concise.

CODE:
{code}

REVIEW:
"""

def generate_one_stream(code: str, max_new_tokens=180, temperature=0.8, top_p=0.9):
    """Yield tokens for a single code review in real time."""
    tok, mdl = _load()
    prompt = PROMPT.format(code=code)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to model device
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "streamer": streamer,
    }

    thread = threading.Thread(target=mdl.generate, kwargs=gen_kwargs)
    thread.start()

    try:
        for piece in streamer:
            yield piece
    finally:
        thread.join()  # Ensure thread cleanup

def generate_candidates(code: str, k=3, max_new_tokens=180, temperature=0.8, top_p=0.9):
    """Generate k candidate reviews for ranking."""
    tok, mdl = _load()
    prompt = PROMPT.format(code=code)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to model device
    out = []

    for _ in range(k):
        try:
            seq = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )
            txt = tok.decode(seq[0], skip_special_tokens=True)
            out.append(txt.split("REVIEW:")[-1].strip())
        except Exception as e:
            print(f"Warning: Failed to generate candidate: {str(e)}")
            continue
    return out

def select_best(scored):
    """Select the best candidate based on score."""
    if not scored:
        raise ValueError("No candidates to select from")
    return max(scored, key=lambda x: x["total"])

def log_event(event: dict, path="logs/events.jsonl"):
    """Log an event to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    event["ts"] = time.time()
    with jsonlines.open(path, "a") as writer:
        writer.write(event)
