"""Benzinga articles per ticker from a licensed Benzinga news API, one JSONL file per year.

    GET <BENZINGA_API_URL>?tickers=<TICKER>&published.gte=...&published.lte=...
    Authorization: Bearer <BENZINGA_API_KEY>; pages are followed through response['next_url'].

The endpoint and key of your Benzinga news subscription are read from the environment variables
BENZINGA_API_URL and BENZINGA_API_KEY. Existing year files are skipped, so an interrupted download can be
resumed. The articles are licensed and must not be redistributed.

Usage:
    python src/data_collection/benzinga.py --tickers-file data/tickers.txt
"""
import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

PAGE_LIMIT = 1000


def get(url, key, ctx, retries=5):
    """GET one page; retries on rate limits and server errors."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 429:
                time.sleep(2 ** attempt); continue
            if e.code >= 500:
                time.sleep(1.5 * (attempt + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed ({last})")


def fetch_year(base, ticker, year, start, end, key, ctx):
    params = {"tickers": ticker, "published.gte": max(f"{year}-01-01", start),
              "published.lte": min(f"{year}-12-31", end), "sort": "published.asc", "limit": PAGE_LIMIT}
    url, out = base + "?" + urllib.parse.urlencode(params), []
    while url:
        data = get(url, key, ctx)
        recs = data.get("results") or []
        out.extend(recs)
        url = data.get("next_url") if recs else None
        if url:
            time.sleep(0.05)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tickers-file", default=C.TICKERS_FILE)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out-dir", default=C.BENZINGA_DIR)
    ap.add_argument("--no-verify-tls", action="store_true", help="only for broken local certificate stores")
    args = ap.parse_args()
    base = os.environ.get("BENZINGA_API_URL", "").strip()
    key = os.environ.get("BENZINGA_API_KEY", "").strip()
    if not base or not key:
        raise SystemExit("set the BENZINGA_API_URL and BENZINGA_API_KEY environment variables")
    ctx = ssl.create_default_context()
    if args.no_verify_tls:
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    for tk in C.read_tickers(args.tickers_file):
        os.makedirs(os.path.join(args.out_dir, tk), exist_ok=True)
        for year in range(int(args.start[:4]), int(args.end[:4]) + 1):
            out = os.path.join(args.out_dir, tk, f"{tk}_{year}.jsonl")
            if os.path.exists(out):
                continue
            try:
                recs = fetch_year(base, tk, year, args.start, args.end, key, ctx)
            except (RuntimeError, urllib.error.HTTPError) as e:
                print(f"  [{tk}] {year} FAILED: {e}", flush=True)
                continue
            with open(out + ".tmp", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
            os.replace(out + ".tmp", out)
            print(f"  [{tk}] {year}: {len(recs):,} articles", flush=True)


if __name__ == "__main__":
    main()
