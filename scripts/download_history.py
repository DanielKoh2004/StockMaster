# scripts/download_history.py
import pandas as pd
from pathlib import Path
import yfinance as yf
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ASSET_CSV = DATA_DIR / "assets.csv"
START_DATE = "2018-01-01"   # adjust as needed
END_DATE = None             # None => up-to-today

def read_assets_tolerant(path: Path):
    """
    Read asset CSV tolerant to commas inside the 'name' field.
    Each valid line must have at least 2 commas separating: class,ticker,name(remaining text)
    Returns list of dicts: {'asset_class','ticker','name'}
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    assets = []
    bad_lines = []
    with path.open("r", encoding="utf-8") as f:
        # read header
        header = f.readline()
        # allow header to be present or absent. We'll attempt to detect header columns.
        # if header doesn't contain 'ticker' assume it's actual first data row
        if 'ticker' not in header.lower() and ',' in header:
            # treat header as data (reset to start)
            f.seek(0)
        for i, raw in enumerate(f, start=2):  # starting at 2 because header line is line 1
            line = raw.strip()
            if not line:
                continue
            # split into 3 parts max: class, ticker, name (name can contain commas)
            parts = line.split(',', 2)
            if len(parts) < 3:
                bad_lines.append((i, line))
                continue
            asset_class, ticker, name = parts[0].strip(), parts[1].strip(), parts[2].strip().strip('"')
            if not ticker:
                bad_lines.append((i, line))
                continue
            assets.append({'asset_class': asset_class or 'Stocks', 'ticker': ticker, 'name': name})
    return assets, bad_lines

def download_ticker_csv(ticker, start=START_DATE, end=END_DATE):
    print("Downloading", ticker)
    try:
        tk = yf.Ticker(ticker)
        if start:
            df = tk.history(start=start, end=end, auto_adjust=False)
        else:
            df = tk.history(period="max", auto_adjust=False)
        if df is None or df.empty:
            print("  no data for", ticker)
            return
        df = df.reset_index()
        fn = f"{ticker}.csv".replace("/", "_")
        out = DATA_DIR / fn
        df.to_csv(out, index=False)
        print("  saved", out)
    except Exception as e:
        print("  ERROR", ticker, e)

def main():
    try:
        assets, bad = read_assets_tolerant(ASSET_CSV)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if bad:
        print("Warning: malformed lines found in assets.csv (line,raw):")
        for ln, raw in bad:
            print(f"  line {ln}: {raw}")
        print("Please fix or quote problematic lines. Proceeding with valid entries...\n")

    tickers = [a['ticker'] for a in assets]
    print(f"Found {len(tickers)} valid tickers to download.")
    for i, tk in enumerate(tickers):
        download_ticker_csv(tk)
        time.sleep(1.0)  # polite pause

if __name__ == "__main__":
    main()
