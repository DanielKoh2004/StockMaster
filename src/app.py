# src/app.py
"""
Top-nav sticky Streamlit app with unique widget keys to avoid duplicate ID errors.
Run from project root:
    python -m streamlit run src/app.py
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from io import BytesIO
from PIL import Image
import requests
import yfinance as yf
import joblib
import time

# local modules (must be importable when run from project root)
from src.data import download_ticker
from src.labeling import create_class_labels, create_reg_target
from src.features import add_basic_features
from src.model import train_xgb_classifier, train_xgb_regressor, train_ann_classifier, train_ann_regressor, load_model, predict_with_model
from src.model import train_nb_classifier, train_svm_classifier, train_svm_regressor
import numpy as np
import pandas as pd
def load_local_combined():
    """Load historical_all.csv into memory."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "historical_all.csv"
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path, parse_dates=["Date"])
    return df


def get_ticker_data(ticker, start="2018-01-01", end=None):
    """Return only rows for the given ticker from the combined CSV."""
    df = load_local_combined()
    if df is None or df.empty:
        return pd.DataFrame()

    df = df[df["Ticker"] == ticker].copy()
    df = prepare_df_numeric(df)

    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]

    return df
def build_asset_map_from_combined():
    """Dynamically build asset groups (stocks, metals, crypto) from combined CSV, with names."""
    df = load_local_combined()
    if df is None or df.empty or "Ticker" not in df.columns:
        return {"Stocks": {}, "Metals": {}, "Cryptocurrencies": {}}

    tickers = sorted(df["Ticker"].unique())
    asset_map = {"Stocks": {}, "Metals": {}, "Cryptocurrencies": {}}

    for tk in tickers:
        # look up friendly name once
        name = lookup_name(tk)

        if str(tk).endswith("-USD"):
            asset_map["Cryptocurrencies"][tk] = name
        elif "=F" in str(tk):  # futures/metals
            asset_map["Metals"][tk] = name
        else:
            asset_map["Stocks"][tk] = name

    return asset_map



def prepare_df_numeric(df_raw):
    """
    Ensure Date is datetime, flatten MultiIndex columns if present,
    and coerce common numeric columns to numeric safely.
    Returns cleaned df_raw.
    """
    if df_raw is None or df_raw.empty:
        return df_raw

    # 1) Flatten MultiIndex columns if yfinance returned multiple tickers at once
    if isinstance(df_raw.columns, pd.MultiIndex):
        # create names like 'AAPL_Close' or 'Close_AAPL' depending on layout
        new_cols = []
        for col in df_raw.columns:
            # col usually is tuple like ('Close','AAPL') or ('AAPL','Close')
            if len(col) >= 2:
                # prefer putting the field name at the end, e.g. 'AAPL_Close'
                new_cols.append(f"{col[0]}_{col[1]}")
            else:
                new_cols.append(col[0])
        df_raw.columns = new_cols

# 2) Ensure Date column exists and is datetime
    if "Date" in df_raw.columns:
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
    else:
        # If Date is in the index, reset it
        if isinstance(df_raw.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            df_raw = df_raw.reset_index().rename(columns={df_raw.index.name or "index": "Date"})
            df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
        else:
            # try to detect a column that looks like a date
            for cand in ["date", "timestamp", "time"]:
                if cand in df_raw.columns:
                    df_raw = df_raw.rename(columns={cand: "Date"})
                    df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
                    break


    # 3) Safely coerce numeric columns
    candidate_cols = ['Open','High','Low','Close','Volume']
    # if you flattened multiindex, these may be like 'AAPL_Close' etc.
    cols_to_try = set(df_raw.columns)  # set of available column names
    for base in candidate_cols:
        # find matching columns that end with the base name (e.g., 'AAPL_Close' or 'Close_AAPL')
        matches = [c for c in cols_to_try if c == base or c.endswith(f"_{base}") or c.startswith(f"{base}_")]
        for c in matches:
            try:
                # if column cells are lists/arrays (rare), extract first numeric candidate
                if df_raw[c].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                    df_raw[c] = df_raw[c].apply(lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) and len(x)>0 else x)
                df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
            except TypeError:
                # fallback: try extracting scalar with str conversion then numeric
                df_raw[c] = df_raw[c].apply(lambda x: np.nan if x is None else (x if isinstance(x, (int,float)) else str(x)))
                df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

    # 4) Final sanity debug (optional) — remove or comment out if noisy
    # print column dtypes to help debugging in future
    # st.write("DEBUG dtypes:", df_raw.dtypes.to_dict())  # uncomment if you want visible debug in app

    return df_raw


BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

st.set_page_config(layout="wide", page_title="StockMaster", initial_sidebar_state="collapsed")


# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def lookup_name(ticker: str) -> str:
    """Return the company/asset name from Yahoo Finance, fallback = ticker."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker
def fetch_logo_img(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        url = info.get("logo_url") or info.get("logo") or info.get("image")
        name = info.get("longName") or info.get("shortName")
        if url and isinstance(url, str) and url.startswith("http"):
            r = requests.get(url, timeout=6)
            img = Image.open(BytesIO(r.content)).convert("RGBA")
            return img, name
        # Fallback: use Clearbit logo API for common US stocks
        clearbit_map = {
            # Stocks
            "AAPL": "apple.com",
            "AMZN": "amazon.com",
            "BAC": "bofa.com",
            "CRM": "salesforce.com",
            "GOOGL": "google.com",
            "GOOG": "google.com",
            "INTC": "intel.com",
            "JPM": "jpmorganchase.com",
            "KO": "coca-cola.com",
            "MA": "mastercard.com",
            "META": "facebook.com",
            "MSFT": "microsoft.com",
            "NFLX": "netflix.com",
            "NKE": "nike.com",
            "NVDA": "nvidia.com",
            "ORCL": "oracle.com",
            "PFE": "pfizer.com",
            "PG": "pg.com",
            "PYPL": "paypal.com",
            "TSLA": "tesla.com",
            "V": "visa.com",
            # Crypto (use project domains)
            "BTC-USD": "bitcoin.org",
            "ETH-USD": "ethereum.org",
            "BNB-USD": "binance.com",
            "ADA-USD": "cardano.org",
            "XRP-USD": "ripple.com",
            "SOL-USD": "solana.com",
            "DOT-USD": "polkadot.network",
            "LTC-USD": "litecoin.org",
            # Metals/Futures (use exchange or info site)
            "GC=F": "cmegroup.com",
            "SI=F": "cmegroup.com",
            "HG=F": "cmegroup.com",
            "NG=F": "cmegroup.com",
            "CL=F": "cmegroup.com",
            "PA=F": "cmegroup.com",
            "PL=F": "cmegroup.com",
        }
        domain = clearbit_map.get(ticker.upper())
        if domain:
            clearbit_url = f"https://logo.clearbit.com/{domain}"
            try:
                r = requests.get(clearbit_url, timeout=6)
                if r.status_code == 200:
                    img = Image.open(BytesIO(r.content)).convert("RGBA")
                    return img, name
            except Exception:
                pass
        return None, name
    except Exception:
        return None, None

def fmt(v):
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return v

def safe_latest_model(prefix, ticker, horizon: int = None):
    """
    Find latest model file for ticker and optional horizon.
    Models trained with horizon should be saved like: xgb_reg_{TICKER}_h{H}.joblib
    """
    if horizon is not None:
        files = sorted(MODELS_DIR.glob(f"{prefix}{ticker}*_h{horizon}*.joblib"))
        if files:
            return files[-1]
    # fallback to any model for ticker
    files = sorted(MODELS_DIR.glob(f"{prefix}{ticker}*.joblib"))
    return files[-1] if files else None

def get_prediction_for_ticker(ticker, mode, horizon=1):
    """
    Attempts to load the latest model for ticker and return (status_string, prediction_value, direction_flag)
    direction_flag: 'up'|'down'|'stable'|'no-model'|'error'
    """
    try:
        df_raw = get_ticker_data(ticker, start="2018-01-01")
        if df_raw.empty or len(df_raw) < 30:
            return ("no-data", None, "no-model")
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        for c in ["Open","High","Low","Close","Volume"]:
            if c in df_raw.columns:
                df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

        if mode.startswith("Class"):
            df = create_class_labels(df_raw, horizon=horizon)
        else:
            df = create_reg_target(df_raw, horizon=horizon)

        df = add_basic_features(df)
        non_features = {"Date","next_close","ret_next","label"}
        features = [c for c in df.columns if c not in non_features and np.issubdtype(df[c].dtype, np.number)]
        if not features:
            return ("no-features", None, "no-model")

        prefix = "xgb_class_" if mode.startswith("Class") else "xgb_reg_"
        model_path = safe_latest_model(prefix, ticker, horizon=horizon)
        if not model_path:
            return ("no-model", None, "no-model")

        model = load_model(model_path)
        last_row = df[features].iloc[[-1]]
        pred = model.predict(last_row)[0]

        if mode.startswith("Class"):
            pred_val = int(pred) - 1
            if pred_val == 1:
                return ("pred-up", pred_val, "up")
            if pred_val == -1:
                return ("pred-down", pred_val, "down")
            return ("pred-stable", pred_val, "stable")
        else:
            pred_price = float(pred)
            current_close = float(df_raw["Close"].iloc[-1])
            dir_flag = "up" if pred_price > current_close else ("stable" if abs(pred_price - current_close) < 1e-9 else "down")
            return ("pred-price", pred_price, dir_flag)
    except Exception as e:
        return ("error", str(e), "error")


def make_price_fig_simple(df_raw, ma_cols=None, preds_plot=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_raw['Date'], y=df_raw['Close'], name='Close', mode='lines', line=dict(width=2)))
    if ma_cols:
        for col in ma_cols:
            if col in df_raw.columns:
                fig.add_trace(go.Scatter(x=df_raw['Date'], y=df_raw[col], name=col.upper(), mode='lines', line=dict(dash='dot'), opacity=0.75))
    if preds_plot is not None:
        fig.add_trace(go.Scatter(x=df_raw['Date'], y=preds_plot, name='Predicted', mode='lines', line=dict(dash='dash', color='firebrick')))
    if 'Volume' in df_raw.columns:
        fig.add_trace(go.Bar(x=df_raw['Date'], y=df_raw['Volume'], name='Volume', marker=dict(color='rgba(120,120,120,0.2)'), yaxis='y2', showlegend=False))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'))
    fig.update_layout(hovermode='x unified', xaxis=dict(rangeslider=dict(visible=True)), height=520, margin=dict(t=20,b=10,l=20,r=20))
    return fig

# ---------------------------
# Clean Sticky Navbar + Fade-In Page Transition
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown("""
<style>
/* Navbar styling */
.navbar {
    position: sticky;
    top: 0;
    background: white;
    z-index: 9999;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #eee;
    font-family: 'Segoe UI', sans-serif;
}

/* Nav item base */
.nav-item > button {
    background: none !important;
    border: none !important;
    color: #333 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px;
    padding: 8px 18px !important;
    transition: all 0.3s ease;
    border-radius: 6px;
}

/* Hover effect */
.nav-item > button:hover {
    color: #0b6efd !important;
    background-color: rgba(11,110,253,0.05) !important;
}

/* Active state */
.nav-active > button {
    color: #0b6efd !important;
    position: relative;
    font-weight: 700 !important;
}

/* Smooth underline animation */
.nav-active > button::after {
    content: '';
    position: absolute;
    bottom: 0px;
    left: 50%;
    transform: translateX(-50%);
    width: 70%;
    height: 2px;
    background: #0b6efd;
    border-radius: 2px;
}
.nav-item > button:hover::after {
    width: 100%;
}

/* 🔥 Fade-in animation for the page */
.fade-in {
    animation: fadeEffect 0.6s ease;
}
@keyframes fadeEffect {
    from {opacity: 0; transform: translateY(10px);}
    to   {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# Menu with icons
menu_items = {
    "Home": "🏠",
    "Predictor": "📈",
    "Train": "⚙️",
    "Assets": "💹",
    "Models": "🗂️",
    "About": "ℹ️"
}


# Render navbar with STOCKMASTER branding
st.markdown('<div style="display:flex;align-items:center;gap:32px;"><span style="font-size:1.6rem;font-weight:900;letter-spacing:2px;color:#222;font-family:Segoe UI, sans-serif;">STOCKMASTER</span>', unsafe_allow_html=True)
st.markdown('<div class="navbar" style="flex:1">', unsafe_allow_html=True)
cols = st.columns(len(menu_items))
for i, (name, icon) in enumerate(menu_items.items()):
    active_class = "nav-active" if st.session_state.page == name else "nav-item"
    with cols[i]:
        st.markdown(f'<div class="{active_class}">', unsafe_allow_html=True)
        if st.button(f"{icon} {name}", key=f"nav_{name}"):
            st.session_state.page = name
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

PAGE = st.session_state.page

# Apply fade-in class wrapper for the whole content below navbar
st.markdown('<div class="fade-in">', unsafe_allow_html=True)



# ---------------------------
# HOME (ranked predictions)
# ---------------------------
if PAGE == "Home":

    st.title("The only stock prediction app you need!")
    st.markdown("""
    Welcome — this page shows assets predicted to go **Up** or **Down** next day.
    Ranking is by predicted percentage difference (predicted_next_close vs current close).
    """)
    st.markdown("---")

    # Model selection for home page
    home_model_type = st.selectbox("Model", ["XGBoost", "ANN", "SVM"], key="home_model_type")

    # Define assets (reuse same lists you used elsewhere)
    asset_map = build_asset_map_from_combined()

    # Choose which asset groups to include on home
    include = st.multiselect("Include asset classes", options=list(asset_map.keys()), default=list(asset_map.keys()), key="home_include")

    # flatten tickers to evaluate
    tickers_to_check = []
    for g in include:
        tickers_to_check += list(asset_map[g].keys())

    st.info(f"Scanning {len(tickers_to_check)} instruments for predictions using {home_model_type} (regression preferred).")

    def compute_pred_for_ticker(ticker, horizon=1):
        out = {"ticker": ticker, "name": None, "current": None, "pred": None, "pct": None, "model": None, "class_dir": None}
        try:
            name = None
            for g in asset_map.values():
                if ticker in g:
                    name = g[ticker]
                    break
            out["name"] = name
            df_raw = get_ticker_data(ticker, start="2018-01-01")
            if df_raw is None or df_raw.empty:
                return out
            df_raw = prepare_df_numeric(df_raw)
            if 'Date' in df_raw.columns:
                df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
            if 'Close' not in df_raw.columns:
                close_cols = [c for c in df_raw.columns if str(c).endswith("_Close") or str(c).startswith("Close_")]
                if close_cols:
                    df_raw = df_raw.rename(columns={close_cols[0]: "Close"})
                else:
                    return out
            df_raw['Close'] = pd.to_numeric(df_raw['Close'], errors='coerce')
            if df_raw['Close'].dropna().empty:
                return out
            current = float(df_raw['Close'].iloc[-1])
            out['current'] = current

            # Model prefix logic
            if home_model_type == "XGBoost":
                reg_prefix = "xgb_reg_"
                class_prefix = "xgb_class_"
            elif home_model_type == "ANN":
                reg_prefix = "ann_reg_"
                class_prefix = "ann_class_"
            elif home_model_type == "SVM":
                reg_prefix = "svm_reg_"
                class_prefix = "svm_class_"
            else:
                reg_prefix = "xgb_reg_"
                class_prefix = "xgb_class_"

            # Try regression model first
            reg_path = safe_latest_model(reg_prefix, ticker, horizon=horizon)
            if reg_path:
                out['model'] = reg_path.name
                df = create_reg_target(df_raw, horizon=horizon)
                df = add_basic_features(df)
                non_features = {"Date", "next_close"}
                features = [c for c in df.columns if c not in non_features and np.issubdtype(df[c].dtype, np.number)]
                if not features:
                    return out
                model = load_model(reg_path)
                last_row = df[features].iloc[[-1]]
                pred = float(model.predict(last_row)[0])
                out['pred'] = pred
                out['pct'] = (pred - current) / current * 100.0 if current != 0 else None
                return out

            # If no reg model, try classification to get direction
            class_path = safe_latest_model(class_prefix, ticker, horizon=horizon)
            if class_path:
                out['model'] = class_path.name
                df = create_class_labels(df_raw, horizon=horizon)
                df = add_basic_features(df)
                non_features = {"Date", "next_close", "ret_next", "label"}
                features = [c for c in df.columns if c not in non_features and np.issubdtype(df[c].dtype, np.number)]
                if features:
                    model = load_model(class_path)
                    last_row = df[features].iloc[[-1]]
                    pred = model.predict(last_row)[0]
                    pred_val = int(pred) - 1
                    out['class_dir'] = "up" if pred_val == 1 else ("down" if pred_val == -1 else "stable")
                return out

            return out
        except Exception as e:
            out['error'] = str(e)
            return out

    # compute for all tickers
    results = []
    for tk in tickers_to_check:
        res = compute_pred_for_ticker(tk)
        results.append(res)


    # Merge regression and classification for each ticker
    # Always show both if available
    top_n = st.number_input("How many top items to show per list", min_value=1, max_value=50, value=3, key="home_topn")

    df_all = pd.DataFrame(results)
    # Adjust regression percentage to be vs last week/month depending on horizon
    def get_pct_vs_horizon(row):
        # Only for regression results
        if row.get('pred') is None or row.get('current') is None:
            return None
        # Try to infer horizon from model name (e.g. xgb_reg_AAPL_h5.joblib)
        import re
        horizon = 1
        model_name = row.get('model') or ''
        m = re.search(r'_h(\d+)', model_name)
        if m:
            horizon = int(m.group(1))
        # Use current as last close, estimate previous by shifting horizon days back
        # This requires access to the price history, so fallback to original pct if not available
        try:
            df_raw = get_ticker_data(row['ticker'], start="2018-01-01")
            df_raw = prepare_df_numeric(df_raw)
            df_raw = df_raw.sort_values('Date')
            if 'Close' in df_raw.columns and len(df_raw) > horizon:
                last = float(df_raw['Close'].iloc[-1])
                prev = float(df_raw['Close'].iloc[-(horizon+1)])
                if prev != 0:
                    return (row['pred'] - prev) / prev * 100.0
        except Exception:
            pass
        return row.get('pct')

    df_all['pct'] = df_all.apply(get_pct_vs_horizon, axis=1)
    # Add a column for direction (from regression or classification)
    df_all['direction'] = df_all.apply(lambda r: r['class_dir'] if r.get('class_dir') else ("up" if r.get('pct', 0) > 0 else ("down" if r.get('pct', 0) < 0 else "stable")), axis=1)

    st.subheader("Potential Upward Insights")
    df_up = df_all[df_all['direction'] == 'up'].sort_values('pct', ascending=False).reset_index(drop=True)
    if df_up.empty:
        st.write("No UP predictions available yet.")
    else:
        # Header row
        header_cols = st.columns([2.5, 1.5, 3])
        with header_cols[0]:
            st.markdown("**Stock Name**")
        with header_cols[1]:
            st.markdown("**5-day Change**")
        with header_cols[2]:
            st.markdown("<div style='text-align:center;font-weight:bold'>Predicted % Change For the Next 5 Days</div>", unsafe_allow_html=True)
        # Data rows
        for idx, row in df_up.head(top_n).iterrows():
            friendly_name = (
                asset_map["Stocks"].get(row['ticker'])
                or asset_map["Metals"].get(row['ticker'])
                or asset_map["Cryptocurrencies"].get(row['ticker'])
                or row['name']
                or row['ticker']
            )
            logo, _ = fetch_logo_img(row['ticker'])
            # 5-day change calculation
            try:
                df_raw = get_ticker_data(row['ticker'], start="2018-01-01")
                df_raw = prepare_df_numeric(df_raw)
                df_raw = df_raw.sort_values('Date')
                if 'Close' in df_raw.columns and len(df_raw) > 5:
                    last = float(df_raw['Close'].iloc[-1])
                    prev5 = float(df_raw['Close'].iloc[-6])
                    change5 = (last - prev5) / prev5 * 100 if prev5 != 0 else None
                else:
                    change5 = None
            except Exception:
                change5 = None
            pct_disp = f"{row['pct']:.2f}%" if row.get('pct') is not None else ""
            cols = st.columns([0.7, 1.8, 1.5, 3])
            with cols[0]:
                if logo:
                    st.image(logo, width=32)
                else:
                    st.write("")
            with cols[1]:
                st.markdown(f"**{friendly_name}**")
            with cols[2]:
                st.markdown(f"{change5:+.2f}%" if change5 is not None else "N/A")
            with cols[3]:
                st.markdown(f"<div style='text-align:center;color:green;font-weight:700'>{pct_disp}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Potential Downward Insights")
    df_down = df_all[df_all['direction'] == 'down'].sort_values('pct', ascending=True).reset_index(drop=True)
    if df_down.empty:
        st.write("No DOWN predictions available.")
    else:
        # Header row
        header_cols = st.columns([2.5, 1.5, 3])
        with header_cols[0]:
            st.markdown("**Stock Name**")
        with header_cols[1]:
            st.markdown("**5-day Change**")
        with header_cols[2]:
            st.markdown("<div style='text-align:center;font-weight:bold'>Predicted % Change For the Next 5 Days</div>", unsafe_allow_html=True)
        # Data rows
        for idx, row in df_down.head(top_n).iterrows():
            friendly_name = (
                asset_map["Stocks"].get(row['ticker'])
                or asset_map["Metals"].get(row['ticker'])
                or asset_map["Cryptocurrencies"].get(row['ticker'])
                or row['name']
                or row['ticker']
            )
            logo, _ = fetch_logo_img(row['ticker'])
            try:
                df_raw = get_ticker_data(row['ticker'], start="2018-01-01")
                df_raw = prepare_df_numeric(df_raw)
                df_raw = df_raw.sort_values('Date')
                if 'Close' in df_raw.columns and len(df_raw) > 5:
                    last = float(df_raw['Close'].iloc[-1])
                    prev5 = float(df_raw['Close'].iloc[-6])
                    change5 = (last - prev5) / prev5 * 100 if prev5 != 0 else None
                else:
                    change5 = None
            except Exception:
                change5 = None
            pct_disp = f"{row['pct']:.2f}%" if row.get('pct') is not None else ""
            cols = st.columns([0.7, 1.8, 1.5, 3])
            with cols[0]:
                if logo:
                    st.image(logo, width=32)
                else:
                    st.write("")
            with cols[1]:
                st.markdown(f"**{friendly_name}**")
            with cols[2]:
                st.markdown(f"{change5:+.2f}%" if change5 is not None else "N/A")
            with cols[3]:
                st.markdown(f"<div style='text-align:center;color:red;font-weight:700'>{pct_disp}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Ranking uses regression model predicted next-close and/or classification direction when available. Both are shown if present.")


# ---------- Predictor page ----------
if PAGE == "Predictor":

    st.title("Predictor")
    st.markdown("Choose asset type and tickers to inspect interactive charts and model predictions.")

    asset_map = build_asset_map_from_combined()

    # Model selection for prediction
    pred_model_type = st.selectbox("Model", ["XGBoost", "ANN", "Naive Bayes", "SVM"], key="pred_model_type")

    # Step 1: Choose asset type
    asset_type = st.selectbox("Asset type", list(asset_map.keys()), key="pred_asset_type_select")

    # Step 2: Choose ticker(s) only from available list
    available = asset_map[asset_type]
    tickers = st.multiselect(
        "Choose tickers",
        options=list(available.keys()),
        format_func=lambda x: f"{x} — {available[x]}",
        default=list(available.keys())[:2],  # first 2 as default
        key="pred_tickers_select"
    )

    # timeframe / horizon control
    horizon_map = {"1 day": 1, "1 week (~5 trading days)": 5, "1 month (~21 trading days)": 21}
    horizon_choice = st.selectbox("Prediction horizon", list(horizon_map.keys()), index=0, key="pred_horizon_select")
    horizon = horizon_map[horizon_choice]

    start = st.text_input("Start date (YYYY-MM-DD)", "2018-01-01", key="pred_start")
    end = st.text_input("End date (YYYY-MM-DD or empty)", "", key="pred_end")
    mode = st.selectbox("Task", ["Classification (direction)", "Regression (price)"], key="pred_mode")
    btn_run = st.button("Load & Show", key="pred_run")

    if btn_run:
        for tk in tickers:
            st.markdown("---")

            # Top row: left = logo+title, center = horizontal metrics, right = small model status
            top_left, metrics_row, top_right = st.columns([0.8, 3.5, 1.2])

            # --- Left: logo + title
            with top_left:
                logo, fullname = fetch_logo_img(tk)
                if logo:
                    st.image(logo, width=64)
                st.subheader(f"{tk}  {fullname or ''}")
                st.caption(f"{asset_type} — Horizon: {horizon_choice}")

            # --- Load and clean data
            df_raw = get_ticker_data(tk, start=start, end=end if end else None)

            if df_raw is None or df_raw.empty:
                st.error(f"No data for {tk}")
                continue
            df_raw = prepare_df_numeric(df_raw)

            # prepare features (pass horizon)
            if mode.startswith("Class"):
                df = create_class_labels(df_raw, horizon=horizon)
            else:
                df = create_reg_target(df_raw, horizon=horizon)
            df = add_basic_features(df)
            if df is None or df.empty:
                st.error(f"No valid data for {tk}")
                continue

            # Safe access
            last_close = df['Close'].iloc[-1] if 'Close' in df.columns else None
            ma5_val = df['ma5'].iloc[-1] if 'ma5' in df.columns else None
            ma10_val = df['ma10'].iloc[-1] if 'ma10' in df.columns else None

            # Next prediction (pass horizon)
            # Choose model prefix based on user selection
            if pred_model_type == "XGBoost":
                prefix = "xgb_class_" if mode.startswith("Class") else "xgb_reg_"
            elif pred_model_type == "ANN":
                prefix = "ann_class_" if mode.startswith("Class") else "ann_reg_"
            elif pred_model_type == "Naive Bayes":
                prefix = "nb_class_"
            elif pred_model_type == "SVM":
                prefix = "svm_class_" if mode.startswith("Class") else "svm_reg_"
            else:
                prefix = "xgb_class_" if mode.startswith("Class") else "xgb_reg_"
            model_path = safe_latest_model(prefix, tk, horizon=horizon)
            # Use correct prefix for display and loading
            if model_path:
                try:
                    model = load_model(model_path)
                    features = [c for c in df.columns if c not in {'Date','next_close','ret_next','label'} and np.issubdtype(df[c].dtype, np.number)]
                    preds = predict_with_model(model, df, features, is_classifier=mode.startswith("Class"))
                    pred_val = preds[-1] if len(preds) else None
                    if mode.startswith("Class"):
                        flag = {1: "up", 0: "stable", -1: "down"}.get(int(pred_val), "stable")
                        status = "ok"
                    else:
                        flag = None
                        status = "ok"
                except Exception:
                    status = "error"
                    pred_val = None
                    flag = None
            else:
                status = "no-model"
                pred_val = None
                flag = None
            # Ensure model name display uses the correct file
            model_display_name = model_path.name if model_path else '—'

            # --- Center: metrics row
            with metrics_row:
                mcols = st.columns([1,1,1,1,1])
                mcols[0].metric("Close", fmt(last_close) if last_close is not None else "N/A")
                mcols[1].metric("MA5", fmt(ma5_val) if ma5_val is not None else "—")
                mcols[2].metric("MA10", fmt(ma10_val) if ma10_val is not None else "—")

                # Tooltip explanations
                pred_explain = {
                    "up": "Up — The model predicts an upward move. Factors: positive momentum, bullish signals, or strong recent performance.",
                    "down": "Down — The model predicts a downward move. Factors: negative momentum, bearish signals, or weak recent performance.",
                    "stable": "Stable — The model predicts little or no change. Factors: neutral signals, low volatility, or lack of clear trend."
                }
                st.markdown("<style>\n.pred-tooltip { position: relative; display: inline-block; cursor: pointer; }\n.pred-tooltip .pred-tooltiptext {\n  visibility: hidden;\n  width: 270px;\n  background-color: #222;\n  color: #fff;\n  text-align: left;\n  border-radius: 6px;\n  padding: 8px 12px;\n  position: absolute;\n  z-index: 1;\n  bottom: 120%;\n  left: 50%;\n  margin-left: -135px;\n  opacity: 0;\n  transition: opacity 0.3s;\n  font-size: 0.98rem;\n}\n.pred-tooltip:hover .pred-tooltiptext {\n  visibility: visible;\n  opacity: 1;\n}\n</style>", unsafe_allow_html=True)

                if status == "no-model":
                    mcols[3].metric("Next", "No model")
                elif status == "no-data":
                    mcols[3].metric("Next", "No data")
                elif status == "error":
                    mcols[3].metric("Next", "Err")
                else:
                    if mode.startswith("Class"):
                        dir_text = {1:"Up",0:"Stable",-1:"Down"}.get(int(pred_val), str(pred_val))
                        color = "🟢" if flag=="up" else ("🔴" if flag=="down" else "⚪")
                        tip = pred_explain.get(flag, "")
                        mcols[3].markdown(f'''<span class="pred-tooltip" style="font-size:1.3rem;font-weight:700;">{color} {dir_text}<span class="pred-tooltiptext">{tip}</span></span>''', unsafe_allow_html=True)
                    else:
                        mcols[3].metric("Next Price", fmt(pred_val))

                mcols[4].write(f"Model: {model_display_name}")

            # --- Right: compact model status/info
            with top_right:
                if model_path:
                    st.success("Model found")
                else:
                    st.info("No model")

            # --- Chart
            preds_plot = None
            if model_path:
                try:
                    model = load_model(model_path)
                    features = [c for c in df.columns if c not in {'Date','next_close','ret_next','label'} and np.issubdtype(df[c].dtype, np.number)]
                    preds = predict_with_model(model, df, features, is_classifier=mode.startswith("Class"))
                    if not mode.startswith("Class"):
                        preds_plot = preds
                except Exception:
                    preds_plot = None

            fig = make_price_fig_simple(df_raw, ma_cols=['ma5','ma10','ma20'], preds_plot=preds_plot)
            st.plotly_chart(fig, use_container_width=True)

            # --- Table
            display_cols = [c for c in ['Date','Close','Open','High','Low','Volume'] if c in df_raw.columns]
            df_disp = df_raw[display_cols].copy().tail(8)
            for c in df_disp.columns:
                if np.issubdtype(df_disp[c].dtype, np.number):
                    df_disp[c] = df_disp[c].map(lambda x: fmt(x))
            st.dataframe(df_disp, use_container_width=True, height=200)
# ---------------------------
# TRAIN (unique keys)
# ---------------------------
if PAGE == "Train":
    st.title("Train Models")
    st.markdown("Train classification or regression XGBoost models and save them in /models/.")

    # train type
    type_sel = st.radio("Train type", ("Classification (direction)", "Regression (price)"), key="train_type")
    # model type
    train_model_type = st.selectbox("Model", ["XGBoost", "ANN", "Naive Bayes", "SVM"], key="train_model_type")

    # asset list
    asset_map = build_asset_map_from_combined()
    asset_classes = list(asset_map.keys())
    asset_sel = st.selectbox("Asset class", asset_classes, key="train_asset")

    tickers = st.multiselect(
        "Select tickers to train",
        options=list(asset_map[asset_sel].keys()),
        format_func=lambda x: f"{x} — {asset_map[asset_sel][x]}",
        default=list(asset_map[asset_sel].keys())[:2],
        key="train_tickers"
    )

    start = st.text_input("Start date", "2018-01-01", key="train_start")
    end = st.text_input("End date (or empty)", "", key="train_end")
    n_splits = st.slider("TimeSeriesSplit folds", 3, 7, 5, key="train_splits")

    # horizon selector for training (names won't collide if you use different horizon)
    horizon_map = {"1 day": 1, "1 week (~5 trading days)": 5, "1 month (~21 trading days)": 21}
    horizon_choice = st.selectbox("Prediction horizon", list(horizon_map.keys()), index=0, key="train_horizon_select")
    horizon = horizon_map[horizon_choice]

    train_btn = st.button("Train now", key="train_run")

    if train_btn:
        if not tickers:
            st.warning("Enter tickers.")
        else:
            # UI elements for progress
            progress_bar = st.progress(0)
            status = st.empty()
            logs_box = st.empty()
            # use a list (mutable) so progress_cb can mutate without nonlocal
            logs = []

            def progress_cb(msg: dict):
                """
                Callback used by src.model.* training functions.
                Expected messages:
                  {'type':'fold_start','fold':i,'n_splits':n}
                  {'type':'fold','fold':i,'n_splits':n,'metrics': {...}}
                  {'type':'done','path': '/full/path/to/model'}
                """
                try:
                    t = msg.get("type")
                    if t == "fold_start":
                        fold = msg.get("fold", 0)
                        n = msg.get("n_splits", 1)
                        pct = int(((fold - 1) / n) * 100)
                        progress_bar.progress(max(0, min(100, pct)))
                        status_text = f"Fold {fold}/{n} — starting..."
                        status.markdown(status_text)
                        logs.insert(0, f"[fold_start] {status_text}")
                        logs_box.text_area("Training logs (most recent first)", value="\n".join(logs), height=260)
                    elif t == "fold":
                        fold = msg.get("fold", 0)
                        n = msg.get("n_splits", 1)
                        metrics = msg.get("metrics", {})
                        pct = int((fold / n) * 100)
                        progress_bar.progress(max(0, min(100, pct)))
                        # Format metrics nicely (float with 4 decimals)
                        mtxt = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])
                        status_text = f"Fold {fold}/{n} done — {mtxt}"
                        status.markdown(status_text)
                        logs.insert(0, f"[fold] {status_text}")
                        logs_box.text_area("Training logs (most recent first)", value="\n".join(logs), height=260)
                    elif t == "done":
                        path = msg.get("path", "")
                        progress_bar.progress(100)
                        status.markdown(f"Saved model: {Path(path).name}")
                        logs.insert(0, f"[done] Saved model: {path}")
                        logs_box.text_area("Training logs (most recent first)", value="\n".join(logs), height=260)
                except Exception as e:
                    logs.insert(0, f"[callback error] {e}")
                    logs_box.text_area("Training logs (most recent first)", value="\n".join(logs), height=260)

            # loop through tickers to train
            for tk in tickers:
                status.markdown(f"Preparing data for {tk} (horizon={horizon})")
                st.write("Training", tk)

                df_raw = get_ticker_data(tk, start=start, end=end if end else None)
                if df_raw is None or df_raw.empty:
                    st.error("No data for " + tk)
                    continue

                df_raw = prepare_df_numeric(df_raw)
                df_raw["Date"] = pd.to_datetime(df_raw["Date"])
                for c in ["Open", "High", "Low", "Close", "Volume"]:
                    if c in df_raw.columns:
                        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

                if type_sel.startswith("Classification"):
                    df = create_class_labels(df_raw, horizon=horizon)
                    df = add_basic_features(df)
                    non_features = {"Date", "next_close", "ret_next", "label"}
                    features = [c for c in df.columns if c not in non_features and np.issubdtype(df[c].dtype, np.number)]
                    if not features:
                        st.error(f"No features available for {tk} after feature engineering.")
                        continue
                    if train_model_type == "XGBoost":
                        save_name = f"xgb_class_{tk}_h{horizon}.joblib"
                        status.markdown(f"Training classifier for {tk} — folds={n_splits}")
                        out = train_xgb_classifier(
                            df, features,
                            label_col="label",
                            save_name=save_name,
                            n_splits=n_splits,
                            progress_callback=progress_cb
                        )
                    elif train_model_type == "ANN":
                        save_name = f"ann_class_{tk}_h{horizon}.joblib"
                        status.markdown(f"Training ANN classifier for {tk} — folds={n_splits}")
                        out = train_ann_classifier(
                            df, features,
                            label_col="label",
                            save_name=save_name,
                            n_splits=n_splits,
                            progress_callback=progress_cb
                        )
                    elif train_model_type == "Naive Bayes":
                        save_name = f"nb_class_{tk}_h{horizon}.joblib"
                        status.markdown(f"Training Naive Bayes classifier for {tk} — folds={n_splits}")
                        out = train_nb_classifier(
                            df, features,
                            label_col="label",
                            save_name=save_name,
                            n_splits=n_splits,
                            progress_callback=progress_cb
                        )
                    elif train_model_type == "SVM":
                        save_name = f"svm_class_{tk}_h{horizon}.joblib"
                        status.markdown(f"Training SVM classifier for {tk} — folds={n_splits}")
                        out = train_svm_classifier(
                            df, features,
                            label_col="label",
                            save_name=save_name,
                            n_splits=n_splits,
                            progress_callback=progress_cb
                        )
                    else:
                        st.error("Unknown model type.")
                        continue
                    st.success(f"Saved classifier: {Path(out).name}")
                else:
                    df = create_reg_target(df_raw, horizon=horizon)
                    df = add_basic_features(df)
                    non_features = {"Date", "next_close"}
                    features = [c for c in df.columns if c not in non_features and np.issubdtype(df[c].dtype, np.number)]
                    if not features:
                        st.error(f"No features available for {tk} after feature engineering.")
                        continue
                    if train_model_type == "XGBoost":
                        save_name = f"xgb_reg_{tk}_h{horizon}.joblib"
                        status.markdown(f"Training regressor for {tk} — folds={n_splits}")
                        out = train_xgb_regressor(
                            df, features,
                            target_col="next_close",
                            save_name=save_name,
                            n_splits=n_splits,
                            progress_callback=progress_cb
                        )
                    elif train_model_type == "ANN":
                        save_name = f"ann_reg_{tk}_h{horizon}.joblib"
                        status.markdown(f"Training ANN regressor for {tk} — folds={n_splits}")
                        out = train_ann_regressor(
                            df, features,
                            target_col="next_close",
                            save_name=save_name,
                            n_splits=n_splits,
                            progress_callback=progress_cb
                        )
                    elif train_model_type == "SVM":
                        save_name = f"svm_reg_{tk}_h{horizon}.joblib"
                        status.markdown(f"Training SVM regressor for {tk} — folds={n_splits}")
                        out = train_svm_regressor(
                            df, features,
                            target_col="next_close",
                            save_name=save_name,
                            n_splits=n_splits,
                            progress_callback=progress_cb
                        )
                    else:
                        st.error("Unknown or unsupported model type for regression.")
                        continue
                    st.success(f"Saved regressor: {Path(out).name}")

            # finalize UI
            progress_bar.progress(100)
            status.markdown("All training completed.")

# ---------------------------
# ASSETS (unique key for choice)
# ---------------------------
if PAGE == "Assets":

    st.title("Assets — All instruments")
    st.markdown("List of available Stocks, Metals and Cryptocurrencies with logos, names, symbols, price and prediction (green/red).")

    asset_map = build_asset_map_from_combined()


    # Model descriptions for tooltip
    model_options = [
        "XGBoost",
        "ANN",
        "SVM",
        "Naive Bayes (classification only)"
    ]
    model_descriptions = {
        "XGBoost": "XGBoost: Gradient boosting, robust for tabular data, handles complex patterns.",
        "ANN": "ANN: Artificial Neural Network, good for non-linear relationships.",
        "SVM": "SVM: Support Vector Machine, effective for high-dimensional spaces.",
        "Naive Bayes (classification only)": "Naive Bayes: classification only, best for simple patterns."
    }
    assets_model_type = st.selectbox(
        "Model",
        model_options,
        key="assets_model_type"
    )
    st.markdown(f"<span style='font-size:0.98rem;color:#888'>{model_descriptions[assets_model_type]}</span>", unsafe_allow_html=True)

    asset_choice = st.selectbox("Which assets to show", ("Stocks","Metals","Cryptocurrencies","All"), key="assets_choice")

    if asset_choice == "All":
        show_list = sum([list(v.keys()) for v in asset_map.values()], [])
    else:
        show_list = list(asset_map[asset_choice].keys())

    # Show spinner while computing predictions
    with st.spinner("Computing predictions for assets..."):
        cols_per_row = 3
        for i, tk in enumerate(show_list):
            df_raw = get_ticker_data(tk, start="2020-01-01")
            logo, fullname = fetch_logo_img(tk)
            price = None
            if not fullname:
                for g in asset_map.values():
                    if tk in g:
                        fullname = g[tk]
                        break
            change5 = None
            if not df_raw.empty:
                df_raw["Date"] = pd.to_datetime(df_raw["Date"])
                df_raw = prepare_df_numeric(df_raw)
                for c in ["Open","High","Low","Close","Volume"]:
                    if c in df_raw.columns:
                        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
                price = df_raw["Close"].iloc[-1]
                # Calculate 5-day change
                if 'Close' in df_raw.columns and len(df_raw) > 5:
                    last = float(df_raw['Close'].iloc[-1])
                    prev5 = float(df_raw['Close'].iloc[-6])
                    if prev5 != 0:
                        change5 = (last - prev5) / prev5 * 100

            # Model prefix logic
            if assets_model_type == "XGBoost":
                reg_prefix = "xgb_reg_"
                class_prefix = "xgb_class_"
            elif assets_model_type == "ANN":
                reg_prefix = "ann_reg_"
                class_prefix = "ann_class_"
            elif assets_model_type == "SVM":
                reg_prefix = "svm_reg_"
                class_prefix = "svm_class_"
            elif assets_model_type.startswith("Naive Bayes"):
                reg_prefix = None
                class_prefix = "nb_class_"
            else:
                reg_prefix = "xgb_reg_"
                class_prefix = "xgb_class_"

            # Get predictions
            flag_r = val_r = None
            flag_c = val_c = None
            # Regression (skip for Naive Bayes)
            if reg_prefix:
                reg_path = safe_latest_model(reg_prefix, tk)
                if reg_path:
                    try:
                        df = create_reg_target(df_raw)
                        df = add_basic_features(df)
                        non_features = {"Date", "next_close"}
                        features = [c for c in df.columns if c not in non_features and np.issubdtype(df[c].dtype, np.number)]
                        if features:
                            model = load_model(reg_path)
                            last_row = df[features].iloc[[-1]]
                            pred = model.predict(last_row)[0]
                            val_r = float(pred)
                            current = float(df_raw["Close"].iloc[-1])
                            flag_r = "up" if val_r > current else ("down" if val_r < current else "stable")
                    except Exception:
                        flag_r = val_r = None
            # Classification
            class_path = safe_latest_model(class_prefix, tk)
            if class_path:
                try:
                    df = create_class_labels(df_raw)
                    df = add_basic_features(df)
                    non_features = {"Date", "next_close", "ret_next", "label"}
                    features = [c for c in df.columns if c not in non_features and np.issubdtype(df[c].dtype, np.number)]
                    if features:
                        model = load_model(class_path)
                        last_row = df[features].iloc[[-1]]
                        pred = model.predict(last_row)[0]
                        pred_val = int(pred) - 1
                        flag_c = "up" if pred_val == 1 else ("down" if pred_val == -1 else "stable")
                        val_c = pred_val
                except Exception:
                    flag_c = val_c = None

            # Compose display: split reg/class, color each line
            reg_line = "No reg model"
            class_line = "No class model"
            reg_color = "gray"
            class_color = "gray"
            # For Naive Bayes, hide regression line
            show_reg = assets_model_type != "Naive Bayes (classification only)"
            if flag_r in ("up", "down", "stable") and val_r is not None:
                reg_line = f"Reg: {flag_r.title()} ({fmt(val_r)})"
                if flag_r == "up":
                    reg_color = "green"
                elif flag_r == "down":
                    reg_color = "red"
            if flag_c in ("up", "down", "stable"):
                class_line = f"Class: {flag_c.title()}"
                if flag_c == "up":
                    class_color = "green"
                elif flag_c == "down":
                    class_color = "red"
            if flag_r not in ("up", "down", "stable") or val_r is None:
                reg_line = "No reg model"
            if flag_c not in ("up", "down", "stable"):
                class_line = "No class model"
            if i % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            col = cols[i % cols_per_row]
            with col:
                friendly = asset_map["Stocks"].get(tk) or asset_map["Metals"].get(tk) or asset_map["Cryptocurrencies"].get(tk) or fullname or tk
                st.markdown(f"<div style='font-size:2rem;font-weight:700;margin-bottom:0.2em'>{friendly}</div>", unsafe_allow_html=True)

                c1, c2 = st.columns([1.1, 1.6], gap="large")
                with c1:
                    st.markdown("<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%'>", unsafe_allow_html=True)
                    if logo:
                        st.image(logo, width=64)
                    else:
                        st.write("")
                    st.markdown(f"<div style='font-size:1.2rem;font-weight:600;margin-top:0.5em'>{tk}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top:0.2em'>Price: <b>{fmt(price) if price is not None else 'N/A'}</b></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top:0.2em'>5d change: <b>{change5:+.2f}%</b></div>" if change5 is not None else "<div style='margin-top:0.2em'>5d change: N/A</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("<style>\n.pred-tooltip { position: relative; display: inline-block; cursor: pointer; }\n.pred-tooltip .pred-tooltiptext {\n  visibility: hidden;\n  width: 270px;\n  background-color: #222;\n  color: #fff;\n  text-align: left;\n  border-radius: 6px;\n  padding: 8px 12px;\n  position: absolute;\n  z-index: 1;\n  bottom: 120%;\n  left: 50%;\n  margin-left: -135px;\n  opacity: 0;\n  transition: opacity 0.3s;\n  font-size: 0.98rem;\n}\n.pred-tooltip:hover .pred-tooltiptext {\n  visibility: visible;\n  opacity: 1;\n}\n</style>", unsafe_allow_html=True)

                    # Tooltip text for regression
                    reg_explain = {
                        "up": "Reg: Up — The regression model predicts the price will rise. Factors: recent upward trend, strong fundamentals, positive news, or technical indicators.",
                        "down": "Reg: Down — The regression model predicts the price will fall. Factors: recent downward trend, weak fundamentals, negative news, or technical indicators.",
                        "stable": "Reg: Stable — The regression model predicts little or no change. Factors: sideways market, low volatility, or mixed signals."
                    }
                    # Tooltip text for classification
                    class_explain = {
                        "up": "Class: Up — The classification model predicts an upward move. Factors: positive momentum, bullish signals, or strong recent performance.",
                        "down": "Class: Down — The classification model predicts a downward move. Factors: negative momentum, bearish signals, or weak recent performance.",
                        "stable": "Class: Stable — The classification model predicts little or no change. Factors: neutral signals, low volatility, or lack of clear trend."
                    }
                    reg_tip = reg_explain.get(flag_r, "No reg model.")
                    class_tip = class_explain.get(flag_c, "No class model.")
                    reg_html = ""
                    if show_reg:
                        reg_html = (
                            f'<span class="pred-tooltip" style="font-size:1.25rem;font-weight:700;color:{reg_color};margin-bottom:0.2em">'
                            f'{reg_line}'
                            f'<span class="pred-tooltiptext">{reg_tip}</span>'
                            f'</span>'
                        )
                    class_html = (
                        f'<span class="pred-tooltip" style="font-size:1.25rem;font-weight:700;color:{class_color}">' 
                        f'{class_line}'
                        f'<span class="pred-tooltiptext">{class_tip}</span>'
                        f'</span>'
                    )
                    html = (
                        "<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%'>"
                        + reg_html
                        + class_html
                        + "</div>"
                    )
                    st.markdown(html, unsafe_allow_html=True)
                st.markdown("<hr style='margin-top:1.2em;margin-bottom:0.8em'>", unsafe_allow_html=True)

# ---------------------------
# MODELS
# ---------------------------
if PAGE == "Models":
    st.title("Saved models")
    st.markdown("Models saved in /models/")
    models = sorted(MODELS_DIR.glob("*.joblib"))
    if not models:
        st.info("No models saved.")
    else:
        for p in models:
            st.write(p.name, "| size KB:", int(p.stat().st_size/1024), "| modified:", time.ctime(p.stat().st_mtime))

# ---------------------------
# ABOUT
# ---------------------------
if PAGE == "About":
    st.title("About")
    st.markdown("This app is a demo for coursework. Predictions are illustrative, not financial advice.")
    st.markdown("- Data source: Yahoo Finance (yfinance)")
    st.markdown("- Models: XGBoost (trained with TimeSeriesSplit)")

# end
