# scripts/compile_data.py
"""Compile per-ticker CSVs in ./data/ into one combined CSV."""

import pandas as pd
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def safe_read_csv(p: Path):
    """Read CSV and normalize Date + Close columns."""
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]

    # Normalize Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)

    # Normalize Close column
    if "Close" not in df.columns:
        for alt in ("Adj Close", "Adj_Close", "AdjClose"):
            if alt in df.columns:
                df = df.rename(columns={alt: "Close"})
                break

    return df

def compile_to_csv(out_name="historical_all.csv"):
    rows = []
    for p in sorted(DATA_DIR.glob("*.csv")):
        ticker = p.stem
        if ticker.lower() in ("assets", "assets_uploaded", "historical_all"):
            continue

        df = safe_read_csv(p)
        if df.empty:
            continue

        wanted = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in df.columns]
        if "Date" not in wanted or "Close" not in wanted:
            print(f"Skipping {ticker} (missing Date/Close)")
            continue

        df = df[wanted].copy()
        df["ticker"] = ticker
        rows.append(df)
        print(f"Added {ticker} rows: {len(df)}")

    if not rows:
        print("No files compiled.")
        return

    big = pd.concat(rows, ignore_index=True)
    big = big.sort_values(["ticker", "Date"])
    out_path = DATA_DIR / out_name
    big.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(big))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="historical_all.csv", help="Output CSV filename")
    args = parser.parse_args()
    compile_to_csv(out_name=args.out)
