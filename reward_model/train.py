# reward_model/train.py
import argparse, os, json, random, math
import jsonlines
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import time
from model import RM

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

KEYWORDS = [
    "rename", "docstring", "type hint", "typing", "exception",
    "error handling", "edge case", "complexity", "pep8", "variable",
    "function", "class", "test", "unit test", "readability"
]

def load_jsonl(path: str) -> List[Dict]:
    items = []
    with jsonlines.open(path) as r:
        for x in r:
            items.append(x)
    return items

def build_examples(raw: List[Dict]) -> List[Dict]:
    ex = []
    for it in raw:
        code = it["code"]
        review = it["review"]
        label = int(it.get("label", 0))
        lint_delta = float(it.get("lint_delta", 0.0))
        tests_passed = float(it.get("tests_passed", 0.0))
        L = len(review.split())
        kw_hits = sum(1 for k in KEYWORDS if k in review.lower())
        ex.append({
            "code": code[:1200],
            "review": review[:1200],
            "label": label,
            "feats": [lint_delta, tests_passed, float(L), float(kw_hits)]
        })
    return ex

def make_pairs(ex: List[Dict]) -> List[Tuple[int,int]]:
    # Pair indices with same code where labels differ (1 > 0)
    by_code = {}
    for i, e in enumerate(ex):
        key = hash(e["code"])
        by_code.setdefault(key, []).append(i)

    pairs = []
    for _k, idxs in by_code.items():
        pos = [i for i in idxs if ex[i]["label"] == 1]
        neg = [i for i in idxs if ex[i]["label"] == 0]
        for i in pos:
            for j in neg:
                pairs.append((i, j))
    random.shuffle(pairs)
    return pairs

def batchify(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_path", default="reward_model/data/seed.jsonl")
    ap.add_argument("--out_dir", default="reward_model")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw = load_jsonl(args.seed_path)
    if not raw:
        raise SystemExit(f"No data in {args.seed_path}")

    examples = build_examples(raw)
    pairs = make_pairs(examples)  # may be empty initially; that's okay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    # Embed all examples once (fast)
    with torch.inference_mode():
        code_emb = st.encode([e["code"] for e in examples], convert_to_tensor=True, normalize_embeddings=True)
        rev_emb  = st.encode([e["review"] for e in examples], convert_to_tensor=True, normalize_embeddings=True)

    feats = torch.tensor([e["feats"] for e in examples], dtype=torch.float32, device=device)
    labels = torch.tensor([e["label"] for e in examples], dtype=torch.float32, device=device)

    rm = RM(dim_text=384, dim_feat=4, hidden=256).to(device)
    opt = optim.AdamW(rm.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    mrk = nn.MarginRankingLoss(margin=args.margin)

    # Simple train loop
    N = len(examples)
    idx = list(range(N))

    for ep in range(args.epochs):
        random.shuffle(idx)
        rm.train()
        total_bce, total_mrk = 0.0, 0.0

        # BCE batches
        for batch in batchify(idx, args.batch_size):
            bc = code_emb[batch]
            br = rev_emb[batch]
            bf = feats[batch]
            y  = labels[batch]
            logits = rm(bc, br, bf)  # (B,)
            loss_bce = bce(logits, y)

            loss = loss_bce
            # Margin ranking on a small random subset of pairs per BCE step (if any)
            if pairs:
                pbatch = random.sample(pairs, k=min(16, len(pairs)))
                i_idx = [i for i, _ in pbatch]
                j_idx = [j for _, j in pbatch]
                s_i = rm(code_emb[i_idx], rev_emb[i_idx], feats[i_idx])
                s_j = rm(code_emb[j_idx], rev_emb[j_idx], feats[j_idx])
                y_rank = torch.ones_like(s_i)  # s_i should be > s_j
                loss_mrk = mrk(s_i, s_j, y_rank)
                loss = loss + 0.5 * loss_mrk
                total_mrk += loss_mrk.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_bce += loss_bce.item()

        print(f"Epoch {ep+1}/{args.epochs} | BCE {total_bce:.3f} | MRK {total_mrk:.3f}")

    # Save weights + small config
    out_pt = Path(args.out_dir) / "rm.pt"
    torch.save(rm.state_dict(), out_pt)
    cfg = {
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "dim_text": 384,
        "dim_feat": 4,
        "hidden": 256,
        "keywords": KEYWORDS,
        "margin": args.margin,
    }
     # ... after saving rm.pt and rm_config.json
    rm_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": os.getenv("GITHUB_SHA", "local"),
        "rm_tag": os.getenv("RM_TAG", "rm@unknown"),
        "auc": float('nan'),   # optional: fill if you compute AUC
        "acc": float('nan'),   # optional: fill if you compute accuracy
        "loss_bce": round(total_bce, 4),
        "loss_rank": round(total_mrk, 4),
    }
    with open(Path(args.out_dir) / "rm_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved: {out_pt} and rm_config.json")

   


if __name__ == "__main__":
    main()
