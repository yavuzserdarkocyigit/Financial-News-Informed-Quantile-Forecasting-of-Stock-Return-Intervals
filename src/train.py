"""Train the quantile Transformer from scratch and write per-stock quantile predictions.

Optimization: AdamW, 5% linear warmup, cosine decay, gradient clipping at 1.0. Checkpoint selection uses
the validation pinball loss, penalized when validation 95% coverage drifts more than 1.5 points from
0.95, so a narrow but under-covering epoch is not selected. Early stopping on the same criterion.

The no-news model (--no-news) is the same model trained with the news block set to zero everywhere.

Usage:
    python src/train.py --seed 7                       # news-aware model   -> run news_s7
    python src/train.py --seed 7 --no-news             # no-news model      -> run nonews_s7
    python src/train.py --data artifacts/dataset_no_sent.pt --arm no-sent --seed 7     # ablation
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config as C
from model import QuantileTransformer, pinball_loss


class Windows(Dataset):
    def __init__(self, blob, mask):
        self.Xp, self.Xn = blob["Xp"][mask], blob["Xn"][mask]
        self.y, self.sid = blob["y"][mask], blob["stock_id"][mask]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"Xp": self.Xp[i].float(), "Xn": self.Xn[i].float(), "y": self.y[i], "sid": self.sid[i]}


def cov95(pred, y):
    return float(((y >= pred[:, 1]) & (y <= pred[:, 5])).float().mean())     # q025 .. q975


def subset_stocks(blob, n):
    """Keep the first n stocks (quick runs and tests)."""
    keep = blob["stock_id"].numpy() < n
    for k in ["Xp", "Xn", "y", "price", "stock_id", "sent_signed"]:
        blob[k] = blob[k][torch.from_numpy(keep)]
    for k in ["dates", "split"]:
        blob[k] = blob[k][keep]
    blob["tickers"] = blob["tickers"][:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=C.DATASET)
    ap.add_argument("--arm", default="", help="run name prefix (default: news, or nonews with --no-news)")
    ap.add_argument("--no-news", action="store_true", help="zero the news block (no-news model)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--stock-dim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--cov-band", type=float, default=0.015, help="tolerated |coverage95 - 0.95|")
    ap.add_argument("--stocks", type=int, default=0, help="use only the first N stocks (quick run)")
    ap.add_argument("--models-dir", default=C.MODELS_DIR)
    ap.add_argument("--preds-dir", default=C.PREDS_DIR)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = C.run_name(args.arm or ("nonews" if args.no_news else "news"), args.seed)

    blob = torch.load(args.data, weights_only=False)
    if args.stocks:
        subset_stocks(blob, args.stocks)
    sid_np = blob["stock_id"].numpy()
    if args.no_news:
        blob["Xn"] = torch.zeros_like(blob["Xn"])

    tr = blob["split"] == "train"; va = blob["split"] == "val"
    dl_tr = DataLoader(Windows(blob, tr), batch_size=args.batch, shuffle=True, drop_last=True)
    dl_va = DataLoader(Windows(blob, va), batch_size=1024)
    model = QuantileTransformer(blob["Xp"].shape[-1], blob["Xn"].shape[-1], len(blob["tickers"]),
                                d_model=args.d_model, n_layers=args.layers, stock_dim=args.stock_dim,
                                dropout=args.dropout, window=blob["window"]).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[train] {run}: stocks={len(blob['tickers'])} train={int(tr.sum()):,} val={int(va.sum()):,} "
          f"params={n_par:,} device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_total = args.epochs * max(len(dl_tr), 1)
    warm = max(int(0.05 * steps_total), 100)

    def lr_at(step):
        if step < warm:
            return args.lr * step / warm
        p = (step - warm) / max(steps_total - warm, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    best, best_state, wait, step = float("inf"), None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train(); run_loss = n = 0
        for b in dl_tr:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            pred = model(b["Xp"].to(device), b["Xn"].to(device), b["sid"].to(device))
            loss = pinball_loss(pred, b["y"].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
            run_loss += loss.item() * len(b["y"]); n += len(b["y"])
        model.eval(); vp, vy = [], []
        with torch.no_grad():
            for b in dl_va:
                vp.append(model(b["Xp"].to(device), b["Xn"].to(device), b["sid"].to(device)).cpu())
                vy.append(b["y"])
        vp, vy = torch.cat(vp), torch.cat(vy)
        vl, cv = float(pinball_loss(vp, vy)), cov95(vp, vy)
        sel = vl * (1.0 + 10.0 * max(0.0, abs(cv - 0.95) - args.cov_band))   # soft coverage penalty
        mark = ""
        if sel < best - 1e-7:
            best, best_state, wait = sel, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
            mark = "  <- best"
        else:
            wait += 1
        print(f"  ep {ep:>3}  train={run_loss / max(n, 1):.5f}  val={vl:.5f}  cov95={cv:.4f}  sel={sel:.5f}{mark}",
              flush=True)
        if wait >= args.patience:
            print(f"[train] early stop at epoch {ep}", flush=True)
            break
    model.load_state_dict(best_state); model.eval()

    os.makedirs(args.models_dir, exist_ok=True)
    mp = os.path.join(args.models_dir, f"{run}.pt")
    torch.save({"state": best_state, "args": vars(args), "tickers": blob["tickers"], "quants": C.QUANTILES}, mp)
    print(f"[train] model -> {mp}", flush=True)

    os.makedirs(args.preds_dir, exist_ok=True)
    with torch.no_grad():
        for si, tk in enumerate(blob["tickers"]):
            m = np.where(sid_np == si)[0]
            outs = []
            for i0 in range(0, len(m), 4096):
                mm = m[i0:i0 + 4096]
                outs.append(model(blob["Xp"][mm].float().to(device), blob["Xn"][mm].float().to(device),
                                  blob["stock_id"][mm].to(device)).cpu().numpy())
            P = np.concatenate(outs)
            df = pd.DataFrame({"Date": blob["dates"][m], "split": blob["split"][m],
                               "y": blob["y"][m].numpy(), "price": blob["price"][m].numpy(),
                               "sent": blob["sent_signed"][m].numpy()})
            for j, c in enumerate(C.QCOLS):
                df[c] = P[:, j]
            df.to_parquet(C.pred_file(args.preds_dir, tk, run), index=False)
    print(f"[train] predictions of {len(blob['tickers'])} stocks -> {args.preds_dir}", flush=True)


if __name__ == "__main__":
    main()
