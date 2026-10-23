"""Zero-shot LLM interval forecasts against the proposed model under matched calibration (Table IX).

A local instruct LLM (fp16, greedy decoding) is asked for the 2.5th and 97.5th percentiles of the upcoming
session's return, once with a compact market summary only and once with the same summary plus that
session's pre-open headlines. Every stock is scored on the same randomly drawn test dates; each method's
raw intervals are corrected with a symmetric split-conformal factor fitted on 100 randomly drawn
validation sessions per stock, and the proposed model's raw intervals get the same correction. Every
method is scored on the sessions where all methods have valid intervals; the table pools these sessions
over stocks (and averages the proposed model over its seeds).

LLM answers are cached per stock, so a run can be resumed and the table recomputed without the LLM.

Usage:
    python src/llm_baseline.py --model Qwen/Qwen3-1.7B
    python src/llm_baseline.py --model google/gemma-3-1b-it
    python src/llm_baseline.py --model meta-llama/Llama-3.2-1B-Instruct --accept-sorted
"""
import argparse
import collections
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd

import config as C
from calibration import winkler
from news_features import MAX_TICKERS, _as_list, _templated, assign_sessions

SYS = ("You are a quantitative analyst. You output ONLY a JSON object giving a 95% prediction "
       "interval for a stock's next-session return, as a percentage change from the last close. "
       "Calibration matters: over many such forecasts the realised return must fall INSIDE the "
       "interval about 95 times out of 100, below it about 2.5 times, and above it about 2.5 "
       "times. An interval that is too narrow is as wrong as one that is too wide.")
MAX_HEADLINES = 40
LC, HC = "q025", "q975"


# --------------------------------------------------------------------------- prompts
def build_prompt(ctx, headlines):
    if headlines:
        hl = headlines[:MAX_HEADLINES]                          # freshest first
        ctx += f"\nPre-open news for the upcoming session ({len(hl)}):\n"
        ctx += "".join(f"[{k}] {t}\n" for k, t in enumerate(hl, 1))
    ctx += ('\nGive the 2.5th and 97.5th percentiles of the return of the UPCOMING session '
            '(percent change from the last close).\n'
            'The interval need not straddle zero: if you expect a directional move, an interval '
            'entirely below or entirely above zero is a valid answer.\n'
            'Respond ONLY as JSON, with no other text: '
            '{"low": <number>, "high": <number>}, where low < high and both are percentages.')
    return ctx


def market_context(tk, preds, dates, prices_dir, vix):
    """One summary block per date, from data through the previous close only: the last ten returns,
    realized volatility (5/20/60 sessions), the average intraday range, relative volume and the VIX."""
    ys = preds["y"].to_numpy(float)
    pos = {d: i for i, d in enumerate(pd.DatetimeIndex(preds["Date"].values))}
    px = pd.read_csv(os.path.join(prices_dir, f"{tk.lower()}.csv"), usecols=["Date", "Close", "High", "Low", "Volume"])
    px["Date"] = pd.to_datetime(px["Date"]).dt.normalize()
    px = px.sort_values("Date").set_index("Date")
    out = {}
    for d in dates:
        hist = ys[max(0, pos[d] - 60):pos[d]] * 100.0
        r10 = np.round(hist[-10:], 2).tolist() if len(hist) else [0.0]
        b = (f"Stock: {tk}\n"
             f"Last {len(r10)} daily returns % (oldest to newest): {', '.join(f'{x:+.2f}' for x in r10)}\n")
        vols = [f"{k}d {np.std(hist[-k:]):.2f}%" for k in (5, 20, 60) if len(hist[-k:]) > 3]
        b += f"Realized daily volatility: {', '.join(vols) if vols else 'n/a'}\n"
        h = px[px.index < d].tail(20)
        if len(h) >= 5:
            b += f"Average intraday high-low range (20 sessions): {float(((h['High'] - h['Low']) / h['Close']).mean() * 100.0):.2f}% of close\n"
            va = float(h["Volume"].mean())
            if va > 0:
                b += f"Last session volume vs its 20-session average: {float(h['Volume'].iloc[-1]) / va:.2f}x\n"
        hv = vix[vix.index < d].tail(20)
        if len(hv) >= 5:
            b += f"VIX: {float(hv.iloc[-1]):.1f} (20-session mean {float(hv.mean()):.1f})\n"
        out[d] = b
    return out


def session_headlines(benzinga_dir, tk, td):
    """date -> headlines of the session's pre-open articles, freshest first (at most 40)."""
    recs = []
    for fp in sorted(glob.glob(os.path.join(benzinga_dir, tk.upper(), f"{tk.upper()}_*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                a = json.loads(ln)
                if _templated(a) or len(_as_list(a.get("tickers"))) > MAX_TICKERS:
                    continue
                title = (a.get("title") or "").strip()
                if title:
                    recs.append({"published": a.get("published"), "title": title})
    if not recs:
        return {}
    df = assign_sessions(pd.DataFrame(recs), td).sort_values("et")
    return {pd.Timestamp(s).normalize(): list(g["title"])[::-1][:40] for s, g in df.groupby("sess")}


# --------------------------------------------------------------------------- LLM
def load_llm(model_id, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    return model.eval(), tok


def generate(prompts, model, tok, device, max_new=96, bs=16, max_len=2048):
    """Greedy completions, batched with left padding."""
    import torch
    tok.padding_side, tok.truncation_side = "left", "left"     # keep the instruction at the end

    def render(p):
        def chat(msgs):
            try:                                               # Qwen3: answer without a reasoning block
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        try:
            return chat([{"role": "system", "content": SYS}, {"role": "user", "content": p}])
        except Exception:                                      # templates without a system role (Gemma)
            return chat([{"role": "user", "content": SYS + "\n\n" + p}])

    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            enc = tok([render(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, do_sample=False)
            out += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
            done = min(i + bs, len(prompts))
            print(f"    [gen] {done}/{len(prompts)}  ({done / max(time.time() - t0, 1e-9):.2f}/s)", end="\r", flush=True)
    print()
    return out


# answers use different key names for the same two numbers
_KEY_PAIRS = [("low", "high"), ("lo", "hi"), ("lower", "upper"), ("lower_bound", "upper_bound"),
              ("lower bound", "upper bound"), ("2.5th percentile", "97.5th percentile"), ("2.5%", "97.5%"),
              ("p2.5", "p97.5"), ("q025", "q975")]


def parse_interval(txt, stats, sort_bounds=False):
    """(low %, high %) from a completion, or None. Crossed bounds are rejected unless sort_bounds."""
    txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.S)
    txt = re.sub(r"^.*?</think>", " ", txt, flags=re.S)
    reason, saw_json = None, False
    for m in re.finditer(r"\{[^{}]*\}", txt, re.S):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        if not isinstance(o, dict):
            continue
        flat = {str(k).strip().lower(): v for k, v in o.items()}
        pair = next(((flat[a], flat[b]) for a, b in _KEY_PAIRS if a in flat and b in flat), None)
        if pair is None:
            nums = [v for v in o.values() if isinstance(v, (int, float))]
            pair = (nums[0], nums[1]) if len(nums) == 2 else None
        if pair is None:
            continue
        try:
            lo, hi = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            continue
        saw_json = True
        if abs(lo) >= 90 or abs(hi) >= 90:
            reason = reason or "out_of_range"; continue
        if lo == hi:
            reason = reason or "zero_width"; continue
        if lo > hi:
            if not sort_bounds:
                reason = reason or "crossed"; continue
            lo, hi = hi, lo
            stats["repaired"] += 1
        stats["ok"] += 1
        return lo, hi
    if not saw_json:                                           # bare numbers, only without any JSON
        nums = [x for x in (float(v) for v in re.findall(r"-?\d+\.?\d*", txt)) if abs(x) < 90]
        if len(nums) >= 2 and nums[0] < nums[1]:
            stats["ok_loose"] += 1
            return nums[0], nums[1]
    stats[reason or "no_json"] += 1
    return None


def elicit(tk, dates, ctx, headlines, model, tok, device, sort_bounds):
    """date -> {'nn': (lo, hi), 'nw': (lo, hi)} log-return bounds without / with news (NaN if unparsed)."""
    prompts, keys = [], []
    for d in dates:
        prompts += [build_prompt(ctx[d], []), build_prompt(ctx[d], headlines.get(pd.Timestamp(d).normalize(), []))]
        keys += [(d, "nn"), (d, "nw")]
    res = {d: {"nn": (np.nan, np.nan), "nw": (np.nan, np.nan)} for d in dates}
    why = collections.Counter()
    for (d, cond), txt in zip(keys, generate(prompts, model, tok, device)):
        got = parse_interval(txt, why, sort_bounds)
        if got:
            res[d][cond] = (float(np.log1p(max(got[0], -95.0) / 100.0)), float(np.log1p(got[1] / 100.0)))
    print(f"  [llm] {tk}: " + ", ".join(f"{k} {v}" for k, v in sorted(why.items())))
    return res


def save_cache(path, val, test):
    rows = [{"Date": d, "split": split, "cond": c, "lo": lo, "hi": hi}
            for split, res in (("val", val), ("test", test)) for d, cm in res.items() for c, (lo, hi) in cm.items()]
    pd.DataFrame(rows).to_parquet(path, index=False)


def load_cache(path):
    if not os.path.exists(path):
        return None, None
    out = {"val": {}, "test": {}}
    for r in pd.read_parquet(path).itertuples():
        out[r.split].setdefault(pd.Timestamp(r.Date), {})[r.cond] = (r.lo, r.hi)
    return out["val"], out["test"]


# --------------------------------------------------------------------------- scoring
def split_conformal(val_lo, val_hi, val_y, test_lo, test_hi, cov=0.95):
    """Symmetric width-normalized split conformal correction fitted on the validation draws."""
    m = np.isfinite(val_lo) & np.isfinite(val_hi) & np.isfinite(val_y)
    if m.sum() < 30:
        return test_lo, test_hi
    w = np.maximum(val_hi[m] - val_lo[m], 1e-6)
    s = np.maximum(val_lo[m] - val_y[m], val_y[m] - val_hi[m]) / w
    q = np.quantile(s, min(1.0, np.ceil((m.sum() + 1) * cov) / m.sum()), method="higher")
    tw = np.maximum(test_hi - test_lo, 1e-6)
    return test_lo - q * tw, test_hi + q * tw


def score(y, lo, hi):
    return {"n": len(y), "cov": float(((y >= lo) & (y <= hi)).mean()), "width": float(np.mean(hi - lo)),
            "wink": float(np.mean(winkler(y, lo, hi, 0.05)))}


def load_run(preds_dir, tk, run):
    p = pd.read_parquet(C.pred_file(preds_dir, tk, run))
    p["Date"] = pd.to_datetime(p["Date"])
    p = p.sort_values("Date").reset_index(drop=True)
    return p, p[p["split"] == "val"].reset_index(drop=True), p[p["split"] == "test"].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="Hugging Face model id")
    ap.add_argument("--tickers-file", default=os.path.join(C.DATA, "llm_tickers.txt"))
    ap.add_argument("--test-days", type=int, default=50, help="test dates shared by all stocks")
    ap.add_argument("--val-days", type=int, default=100, help="validation sessions per stock")
    ap.add_argument("--accept-sorted", action="store_true", help="repair crossed bounds by swapping them")
    ap.add_argument("--cache-only", action="store_true", help="recompute the table from cached answers only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)), help="seeds of the proposed model")
    ap.add_argument("--preds-dir", default=C.PREDS_DIR)
    ap.add_argument("--news-dir", default=C.NEWS_DIR)
    ap.add_argument("--prices-dir", default=C.PRICES_DIR)
    ap.add_argument("--vix-file", default=C.VIX_FILE)
    ap.add_argument("--benzinga-dir", default=C.BENZINGA_DIR)
    ap.add_argument("--cache-root", default=os.path.join(C.ARTIFACTS, "llm_cache"))
    ap.add_argument("--out-dir", default=C.RESULTS_DIR)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    seeds = [int(s) for s in args.seeds.split(",")]
    ref_run = C.run_name("news", seeds[0])
    name = args.model.split("/")[-1] + ("_sorted" if args.accept_sorted else "")
    cache_dir = os.path.join(args.cache_root, f"{args.model.split('/')[-1]}_d{args.test_days}_c{args.val_days}"
                             f"_b0_s{args.seed}" + ("_sorted" if args.accept_sorted else ""))
    os.makedirs(cache_dir, exist_ok=True)
    tickers = [t for t in C.read_tickers(args.tickers_file) if os.path.exists(C.pred_file(args.preds_dir, t, ref_run))]

    # the same test dates for every stock
    sets = [set(load_run(args.preds_dir, tk, ref_run)[2]["Date"]) for tk in tickers]
    inter = sorted(set.intersection(*sets))
    dates = sorted(pd.Timestamp(x) for x in rng.choice(np.array(inter), size=min(args.test_days, len(inter)), replace=False))
    print(f"[llm] {name}: {len(tickers)} stocks x {len(dates)} shared test dates", flush=True)
    vix = pd.read_parquet(args.vix_file)
    vix = vix.assign(Date=pd.to_datetime(vix["Date"]).dt.normalize()).dropna(subset=["Date"]) \
             .set_index("Date")["VIX_Close"].astype(float).sort_index()
    model = tok = device = None

    rows = []
    for tk in tickers:
        p, v, t = load_run(args.preds_dir, tk, ref_run)
        sel = [i for i, d in enumerate(pd.DatetimeIndex(t["Date"].values)) if d in set(dates)]
        tdays = [pd.Timestamp(x) for x in t["Date"].values[sel]]
        vsel = sorted(rng.choice(len(v), size=min(args.val_days, len(v)), replace=False))
        vdays = [pd.Timestamp(x) for x in v["Date"].values[vsel]]

        cpath = os.path.join(cache_dir, f"{tk.lower()}.parquet")
        ans_v, ans_t = load_cache(cpath)
        if ans_v is None or not all(d in ans_v for d in vdays) or not all(d in ans_t for d in tdays):
            if args.cache_only:
                raise SystemExit(f"[llm] {tk}: cached answers missing in {cache_dir}")
            if model is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, tok = load_llm(args.model, device)
            heads = session_headlines(args.benzinga_dir, tk, np.sort(p["Date"].values.astype("datetime64[ns]")))
            ctx = market_context(tk, p, tdays + vdays, args.prices_dir, vix)
            ans_v = elicit(tk, vdays, ctx, heads, model, tok, device, args.accept_sorted)
            ans_t = elicit(tk, tdays, ctx, heads, model, tok, device, args.accept_sorted)
            save_cache(cpath, ans_v, ans_t)

        yv, yt = v["y"].to_numpy(float)[vsel], t["y"].to_numpy(float)[sel]
        methods = {}
        for cond, nm in (("nn", "LLM no news"), ("nw", "LLM news")):
            vlo, vhi = (np.array([ans_v[d][cond][k] for d in vdays]) for k in (0, 1))
            tlo, thi = (np.array([ans_t[d][cond][k] for d in tdays]) for k in (0, 1))
            methods[(nm, "raw", None)] = (tlo, thi)
            methods[(nm, "matched", None)] = split_conformal(vlo, vhi, yv, tlo, thi)
        for s in seeds:
            for arm, nm in (("news", "Proposed (news)"), ("nonews", "Proposed (no news)")):
                _, va, te = load_run(args.preds_dir, tk, C.run_name(arm, s))
                if len(te) != len(t) or len(va) != len(v):
                    continue
                rlo, rhi = te[LC].to_numpy(float)[sel], te[HC].to_numpy(float)[sel]
                methods[(nm, "raw", s)] = (rlo, rhi)
                methods[(nm, "matched", s)] = split_conformal(va[LC].to_numpy(float)[vsel], va[HC].to_numpy(float)[vsel],
                                                              va["y"].to_numpy(float)[vsel], rlo, rhi)
        valid = np.ones(len(yt), bool)                         # score every method on the same sessions
        for lo, hi in methods.values():
            valid &= np.isfinite(lo) & np.isfinite(hi)
        if valid.sum() < 20:
            continue
        for (nm, kind, s), (lo, hi) in methods.items():
            rows.append({"ticker": tk, "method": nm, "kind": kind, "seed": s, **score(yt[valid], lo[valid], hi[valid])})

    d = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    d.to_csv(os.path.join(args.out_dir, f"llm_{name}_per_stock.csv"), index=False)
    for c in ("cov", "width", "wink"):                         # pool the scored sessions of all stocks
        d[f"{c}_x_n"] = d[c] * d["n"]
    pooled = d.groupby(["method", "kind", "seed"], dropna=False)[["n", "cov_x_n", "width_x_n", "wink_x_n"]].sum()
    for c in ("cov", "width", "wink"):
        pooled[c] = pooled[f"{c}_x_n"] / pooled["n"]
    agg = pooled[["n", "cov", "width", "wink"]].groupby(["method", "kind"]).mean()   # then proposed-model seeds
    agg.to_csv(os.path.join(args.out_dir, f"llm_{name}.csv"))
    print(f"\n[llm] {name}: {d['ticker'].nunique()} stocks, raw | matched calibration")
    print(f"{'method':<20} {'sessions':>8} {'PICP':>6} {'width':>7} | {'PICP':>6} {'width':>7} {'Winkler':>8}")
    for nm in ("Proposed (news)", "Proposed (no news)", "LLM news", "LLM no news"):
        r, c = agg.loc[(nm, "raw")], agg.loc[(nm, "matched")]
        print(f"{nm:<20} {int(r['n']):>8} {r['cov']:>6.3f} {r['width']:>7.4f} | {c['cov']:>6.3f} {c['width']:>7.4f} {c['wink']:>8.4f}")


if __name__ == "__main__":
    main()
