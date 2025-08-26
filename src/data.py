import yfinance as yf
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

def download_ticker(ticker="AAPL", start="2018-01-01", end=None, force=False):
    """
    Download OHLCV for a ticker and save CSV under data/.
    Returns a pandas DataFrame with Date column parsed.
    """
    out = DATA_DIR / f"{ticker.upper()}.csv"
    if out.exists() and not force:
        df = pd.read_csv(out, parse_dates=["Date"])
        return df
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns=str)
    df.to_csv(out, index=False)
    return df

def load_csv(ticker):
    out = DATA_DIR / f"{ticker.upper()}.csv"
    if out.exists():
        return pd.read_csv(out, parse_dates=["Date"])
    return pd.DataFrame()
