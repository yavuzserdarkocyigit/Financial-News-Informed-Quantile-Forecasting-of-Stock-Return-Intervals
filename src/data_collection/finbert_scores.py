"""Article-level FinBERT scores (ProsusAI/finbert) in a resumable SQLite database.

Every Benzinga article is scored once, even when it is filed under several tickers. Templated articles,
articles with more than 25 tickers and articles without a title are skipped (news_features.py later keeps
only articles with at most three tickers). Per article:
    title and body class probabilities (negative, neutral, positive) and signed sentiment, where the body
    is the HTML-stripped text, or the title when the body is empty
    a 96-d unit-norm title embedding: the [CLS] state of the last layer, multiplied by a fixed random
    Gaussian projection (seed 1729)
Titles are truncated to 64 tokens and bodies to 512 tokens.

Usage:
    python src/data_collection/finbert_scores.py --tickers-file data/tickers.txt
"""
import argparse
import glob
import html
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  
from news_features import _as_list, _templated  

MODEL = "ProsusAI/finbert"
WEIGHTS_REVISION = "refs/pr/29"                      # same weights as main, in safetensors format
TAG_RE, SPACE_RE = re.compile(r"<[^>]+>"), re.compile(r"\s+")
SCHEMA = """CREATE TABLE IF NOT EXISTS scores (
    benzinga_id TEXT PRIMARY KEY, published TEXT NOT NULL,
    title_negative REAL NOT NULL, title_neutral REAL NOT NULL, title_positive REAL NOT NULL,
    title_sentiment REAL NOT NULL, title_confidence REAL NOT NULL, title_entropy REAL NOT NULL,
    body_negative REAL, body_neutral REAL, body_positive REAL, body_sentiment REAL,
    body_confidence REAL, body_entropy REAL,
    semantic_embedding BLOB NOT NULL, projection_dim INTEGER NOT NULL, scored_at TEXT NOT NULL)"""
INSERT = "INSERT OR REPLACE INTO scores VALUES (" + ",".join("?" * 17) + ")"


def clean_body(value):
    if not isinstance(value, str) or not value:
        return ""
    return SPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", value))).strip()


def articles(benzinga_dir, tickers, done):
    """Unscored, eligible articles of all tickers, each id once."""
    seen = set(done)
    for tk in tickers:
        for path in sorted(glob.glob(os.path.join(benzinga_dir, tk, f"{tk}_*.jsonl"))):
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    try:
                        a = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    key = str(a["benzinga_id"]) if a.get("benzinga_id") is not None else ""
                    title, published = str(a.get("title") or "").strip(), str(a.get("published") or "").strip()
                    if (not key or key in seen or not title or not published or _templated(a)
                            or len(_as_list(a.get("tickers"))) > 25):
                        continue
                    seen.add(key)
                    yield {"id": key, "published": published, "title": title,
                           "body": clean_body(a.get("body")) or title}


def entropy(p):
    p = np.clip(p, 1e-8, 1.0)
    return (-np.sum(p * np.log(p), axis=1) / math.log(3.0)).astype(np.float32)


def infer(texts, model, tok, order, device, max_tokens, embedding):
    import torch
    enc = tok(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.no_grad()
    with torch.no_grad(), amp:
        out = model(**enc, output_hidden_states=embedding)
        prob = torch.softmax(out.logits, dim=-1).index_select(1, order).float().cpu().numpy()
    cls = out.hidden_states[-1][:, 0].float().cpu().numpy() if embedding else None
    return prob.astype(np.float32), cls


def score_batch(batch, model, tok, order, device, projection, title_tokens, body_tokens):
    tp, emb = infer([a["title"] for a in batch], model, tok, order, device, title_tokens, True)
    bp, _ = infer([a["body"] for a in batch], model, tok, order, device, body_tokens, False)
    sem = emb @ projection
    sem = (sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-8)).astype(np.float16)
    ts, bs = tp[:, 2] - tp[:, 0], bp[:, 2] - bp[:, 0]
    te, be = entropy(tp), entropy(bp)
    now = datetime.now(timezone.utc).isoformat()
    return [(a["id"], a["published"], *map(float, tp[i]), float(ts[i]), float(tp[i].max()), float(te[i]),
             *map(float, bp[i]), float(bs[i]), float(bp[i].max()), float(be[i]),
             sqlite3.Binary(sem[i].tobytes()), int(sem.shape[1]), now) for i, a in enumerate(batch)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tickers-file", default=C.TICKERS_FILE)
    ap.add_argument("--benzinga-dir", default=C.BENZINGA_DIR)
    ap.add_argument("--database", default=C.FINBERT_DB)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--title-max-tokens", type=int, default=64)
    ap.add_argument("--body-max-tokens", type=int, default=512)
    ap.add_argument("--projection-dim", type=int, default=96)
    ap.add_argument("--projection-seed", type=int, default=1729)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new articles (quick test)")
    args = ap.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    os.makedirs(os.path.dirname(os.path.abspath(args.database)), exist_ok=True)
    con = sqlite3.connect(args.database, timeout=120.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(SCHEMA)
    done = {r[0] for r in con.execute("SELECT benzinga_id FROM scores")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    # the main branch ships only a pickled checkpoint; the safetensors weights live on a conversion branch
    model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(MODEL))
    state = load_file(hf_hub_download(MODEL, "model.safetensors", revision=WEIGHTS_REVISION))
    state.pop("bert.embeddings.position_ids", None)          # buffer that newer versions no longer store
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    labels = {int(i): str(lab).lower() for i, lab in model.config.id2label.items()}
    order = torch.tensor([next(i for i, lab in labels.items() if lab == name)
                          for name in ("negative", "neutral", "positive")], device=device)
    rng = np.random.default_rng(args.projection_seed)
    projection = rng.normal(size=(model.config.hidden_size, args.projection_dim)).astype(np.float32)
    projection /= np.sqrt(float(args.projection_dim))

    added, batch, t0 = 0, [], time.time()
    for a in articles(args.benzinga_dir, C.read_tickers(args.tickers_file), done):
        batch.append(a)
        if args.limit and added + len(batch) >= args.limit or len(batch) == args.batch_size:
            con.executemany(INSERT, score_batch(batch, model, tok, order, device, projection,
                                                args.title_max_tokens, args.body_max_tokens))
            con.commit()
            added += len(batch); batch = []
            if added % (args.batch_size * 25) == 0:
                print(f"  [finbert] {added:,} articles ({added / (time.time() - t0):.1f}/s)", flush=True)
            if args.limit and added >= args.limit:
                break
    if batch:
        con.executemany(INSERT, score_batch(batch, model, tok, order, device, projection,
                                            args.title_max_tokens, args.body_max_tokens))
        con.commit(); added += len(batch)
    print(f"[finbert] {added:,} new articles -> {args.database}")


if __name__ == "__main__":
    main()
