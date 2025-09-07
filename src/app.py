
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
from src.data import download_ticker
from src.labeling import create_class_labels, create_reg_target
from src.features import add_basic_features
from src.model import (
    train_xgb_classifier, train_xgb_regressor,
    train_ann_classifier, train_ann_regressor,
    load_model, predict_with_model, predict_proba_with_model
)
from src.model import train_nb_classifier, train_svm_classifier, train_svm_regressor

# Small utility to safely rerun Streamlit after updating session_state
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass
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
    """Dynamically build asset groups (stocks, metals, crypto) from combined CSV, with names.

    Excludes certain ambiguous/raw tickers that shouldn't be shown as assets (e.g., BTC spot proxy, GC).
    """
    df = load_local_combined()
    if df is None or df.empty or "Ticker" not in df.columns:
        return {"Stocks": {}, "Metals": {}, "Cryptocurrencies": {}}

    # Exclude ambiguous/raw symbols from UI lists
    EXCLUDE_TICKERS = {"BTC", "GC"}
    tickers = [t for t in sorted(df["Ticker"].unique()) if t not in EXCLUDE_TICKERS]
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

def load_model_metadata(model_path: Path):
    try:
        meta_path = Path(model_path).with_suffix(Path(model_path).suffix + ".meta.json")
        if meta_path.exists():
            import json
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def ensure_reports_dir():
    base = Path(__file__).resolve().parents[1]
    rpt = base / "reports"
    rpt.mkdir(exist_ok=True)
    return rpt

def render_model_card(model, model_path: Path, meta: dict):
    with st.expander("Model card", expanded=False):
        cols = st.columns(3)
        cols[0].markdown(f"**File**: {model_path.name if model_path else '—'}")
        if meta:
            cols[1].markdown(f"**Best CV score**: {meta.get('best_score', '—'):.4f}" if meta.get('best_score') is not None else "**Best CV score**: —")
            cols[2].markdown(f"**Folds**: {meta.get('n_splits','—')}")
            c2 = st.columns(3)
            c2[0].markdown(f"**Train rows**: {meta.get('train_rows','—')}")
            c2[1].markdown(f"**Features**: {meta.get('n_features','—')}")
            c2[2].markdown(f"**Train time**: {int(meta.get('duration_sec',0))}s")
            st.markdown(f"**Window**: {meta.get('date_start','—')} → {meta.get('date_end','—')}")
            bp = meta.get('best_params')
            if bp:
                st.markdown("**Best params**")
                st.json(bp)
        # Fallback to estimator params
        try:
            params = model.get_params()
            # filter common estimator params
            keys = [k for k in params.keys() if any(k.startswith(p) for p in ('xgb__','ann__','svc__','svr__','nb__'))]
            subset = {k: params[k] for k in keys}
            if subset:
                st.markdown("**Estimator params**")
                st.json(subset)
        except Exception:
            pass

def render_confusion_matrix(y_true, y_pred, labels=(-1,0,1), title="Confusion Matrix"):
    try:
        from sklearn.metrics import confusion_matrix
        import plotly.figure_factory as ff
        cm = confusion_matrix(y_true, y_pred, labels=list(labels))
        z = cm.astype(int)
        z_text = [[str(v) for v in row] for row in z]
        fig = ff.create_annotated_heatmap(z, x=[str(l) for l in labels], y=[str(l) for l in labels], colorscale='Blues', showscale=True, annotation_text=z_text)
        fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual", height=320)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Confusion matrix unavailable: {e}")

def threshold_predict_from_proba(probs, threshold=0.5):
    """Given probs shaped (n,3) in order [-1,0,1], return labels using threshold: if P(1)>=t -> 1; elif P(-1)>=t -> -1; else 0."""
    if probs is None:
        return None
    import numpy as np
    p_down = probs[:,0]
    p_stable = probs[:,1]
    p_up = probs[:,2]
    preds = np.where(p_up >= threshold, 1, np.where(p_down >= threshold, -1, 0))
    return preds

def compute_macro_aucs(y_true, probs):
    try:
        import numpy as np
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score, average_precision_score
        classes = [-1,0,1]
        Y = label_binarize(y_true, classes=classes)
        # probs shape (n,3) order [-1,0,1]
        roc_auc = roc_auc_score(Y, probs, average='macro', multi_class='ovr')
        pr_auc = average_precision_score(Y, probs, average='macro')
        return float(roc_auc), float(pr_auc)
    except Exception:
        return None, None

def render_fold_timeline(fold_ranges):
    if not fold_ranges:
        return
    try:
        import pandas as pd
        import plotly.express as px
        data = []
        for i, fr in enumerate(fold_ranges, start=1):
            data.append({"Fold": f"Train {i}", "Start": fr['train'][0], "Finish": fr['train'][1], "Type": "Train"})
            data.append({"Fold": f"Test {i}", "Start": fr['test'][0], "Finish": fr['test'][1], "Type": "Test"})
        df = pd.DataFrame(data)
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Fold", color="Type")
        fig.update_layout(height=220, margin=dict(t=20,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Fallback: simple text list
        lines = []
        for i, fr in enumerate(fold_ranges, start=1):
            lines.append(f"Fold {i} — Train: {fr['train'][0]} → {fr['train'][1]} | Test: {fr['test'][0]} → {fr['test'][1]}")
        st.code("\n".join(lines))

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
    "Compare": "📊",
    "Summary": "📊",
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

def render_data_health_panel():
    df = load_local_combined()
    if df is None or df.empty:
        st.info("Data Health: historical_all.csv is missing or empty in data/.")
        return
    try:
        total_rows = len(df)
        tickers = df['Ticker'].nunique() if 'Ticker' in df.columns else None
        last_date = pd.to_datetime(df['Date']).max() if 'Date' in df.columns else None
        with st.expander("Data Health", expanded=True):
            cols = st.columns(3)
            cols[0].metric("Rows", f"{total_rows:,}")
            cols[1].metric("Tickers", f"{tickers}")
            cols[2].metric("Last date", str(last_date.date()) if last_date is not None else "—")
            st.caption("Source: Yahoo Finance (pre-collected). Use Train/Evaluate with these cached files.")
    except Exception:
        pass



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

    # Brief dataset status snapshot
    render_data_health_panel()

    # Model selection for home page (default to SVM)
    home_model_options = ["XGBoost", "ANN", "SVM"]
    home_model_default_index = home_model_options.index("SVM") if "SVM" in home_model_options else 0
    home_model_type = st.selectbox("Model", home_model_options, index=home_model_default_index, key="home_model_type")

    # Define assets (reuse same lists you used elsewhere)
    asset_map = build_asset_map_from_combined()

    # Choose which asset groups to include on home
    include = st.multiselect("Include asset classes", options=list(asset_map.keys()), default=list(asset_map.keys()), key="home_include")

    # flatten tickers to evaluate
    tickers_to_check = []
    for g in include:
        tickers_to_check += list(asset_map[g].keys())

    st.info(f"Scanning {len(tickers_to_check)} instruments for predictions using {home_model_type} (regression preferred).")

    def compute_pred_for_ticker(ticker, horizon=None):
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

            # Always use horizon=1 for model and features
            fixed_horizon = 1
            reg_path = safe_latest_model(reg_prefix, ticker, horizon=fixed_horizon)
            if reg_path:
                out['model'] = reg_path.name
                df = create_reg_target(df_raw, horizon=fixed_horizon)
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

            class_path = safe_latest_model(class_prefix, ticker, horizon=fixed_horizon)
            if class_path:
                out['model'] = class_path.name
                df = create_class_labels(df_raw, horizon=fixed_horizon)
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
    # Model selection for prediction (default to SVM)
    pred_model_options = ["XGBoost", "ANN", "Naive Bayes", "SVM"]
    pred_model_default_index = pred_model_options.index("SVM") if "SVM" in pred_model_options else 0
    pred_model_type = st.selectbox("Model", pred_model_options, index=pred_model_default_index, key="pred_model_type")

    # Step 1: Choose asset type
    # Default asset type to Stocks when available
    asset_types = list(asset_map.keys())
    asset_type_default_index = asset_types.index("Stocks") if "Stocks" in asset_types else 0
    asset_type = st.selectbox("Asset type", asset_types, index=asset_type_default_index, key="pred_asset_type_select")

    # Step 2: Choose ticker(s) only from available list
    available = asset_map[asset_type]
    # Default tickers for Stocks: BAC, NKE, TSLA, CRM, INTC (fallback to first 2 if not applicable)
    preferred_stock_defaults = ["BAC", "NKE", "TSLA", "CRM", "INTC"]
    default_tickers = [t for t in preferred_stock_defaults if t in available.keys()] if asset_type == "Stocks" else []
    if not default_tickers:
        default_tickers = list(available.keys())[:2]
    tickers = st.multiselect(
        "Choose tickers",
        options=list(available.keys()),
        format_func=lambda x: f"{x} — {available[x]}",
        default=default_tickers,
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
                    # Model card with metadata and params
                    meta = load_model_metadata(model_path)
                    try:
                        model_loaded = load_model(model_path)
                    except Exception:
                        model_loaded = None
                    render_model_card(model_loaded, model_path, meta)
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

            # --- Classification analysis with threshold, metrics, ROC/PR, misclassifications
            if mode.startswith("Class") and model_path:
                try:
                    model = load_model(model_path)
                    features = [c for c in df.columns if c not in {'Date','next_close','ret_next','label'} and np.issubdtype(df[c].dtype, np.number)]
                    probs = predict_proba_with_model(model, df, features)
                    if probs is not None and 'label' in df.columns:
                        st.markdown("---")
                        st.subheader("Classification analysis")
                        thr = st.slider("Decision threshold (Up/Down)", 0.3, 0.8, 0.5, 0.01, key=f"thr_{tk}")
                        preds_thr = threshold_predict_from_proba(probs, threshold=thr)
                        y_true = df['label']
                        # Metrics
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        acc = accuracy_score(y_true, preds_thr)
                        prec = precision_score(y_true, preds_thr, average='weighted', zero_division=0)
                        rec = recall_score(y_true, preds_thr, average='weighted', zero_division=0)
                        f1 = f1_score(y_true, preds_thr, average='weighted', zero_division=0)
                        mcols2 = st.columns(4)
                        mcols2[0].metric("Accuracy", f"{acc:.3f}")
                        mcols2[1].metric("Precision(w)", f"{prec:.3f}")
                        mcols2[2].metric("Recall(w)", f"{rec:.3f}")
                        mcols2[3].metric("F1(w)", f"{f1:.3f}")
                        # Confusion matrix
                        render_confusion_matrix(y_true, preds_thr, title="Confusion (thresholded)")
                        # ROC/PR AUC
                        roc_auc, pr_auc = compute_macro_aucs(y_true, probs)
                        st.caption(f"Macro ROC-AUC: {roc_auc if roc_auc is not None else 'NA'} | Macro PR-AUC: {pr_auc if pr_auc is not None else 'NA'}")
                        # Misclassifications table
                        wrong_idx = (preds_thr != y_true).to_numpy().nonzero()[0]
                        if len(wrong_idx) > 0:
                            last_n = 10
                            sel = wrong_idx[-last_n:]
                            cols_show = ['Date','Close','label'] + [c for c in ['ma5','ma10','rsi14','macd','bb_high','bb_low'] if c in df.columns]
                            mis = df.iloc[sel][cols_show].copy()
                            mis['pred'] = preds_thr[sel]
                            st.markdown("#### Recent misclassifications")
                            st.dataframe(mis, use_container_width=True)
                except Exception as e:
                    st.info(f"Analysis unavailable: {e}")
# ---------------------------
# TRAIN (unique keys)
# ---------------------------
if PAGE == "Train":
    st.title("Train Models")
    st.markdown("Train classification or regression XGBoost models and save them in /models/.")

    # train type
    type_sel = st.radio("Train type", ("Classification (direction)", "Regression (price)"), key="train_type")
    # model type default to SVM
    train_model_options = ["XGBoost", "ANN", "Naive Bayes", "SVM"]
    train_model_default_index = train_model_options.index("SVM") if "SVM" in train_model_options else 0
    train_model_type = st.selectbox("Model", train_model_options, index=train_model_default_index, key="train_model_type")

    # asset list
    asset_map = build_asset_map_from_combined()
    asset_classes = list(asset_map.keys())
    asset_default_index = asset_classes.index("Stocks") if "Stocks" in asset_classes else 0
    asset_sel = st.selectbox("Asset class", asset_classes, index=asset_default_index, key="train_asset")

    # preferred default tickers when in Stocks
    preferred_stock_defaults = ["BAC", "NKE", "TSLA", "CRM", "INTC"]
    available_train = list(asset_map[asset_sel].keys())
    default_train_tickers = [t for t in preferred_stock_defaults if t in available_train] if asset_sel == "Stocks" else available_train[:2]
    tickers = st.multiselect(
        "Select tickers to train",
        options=available_train,
        format_func=lambda x: f"{x} — {asset_map[asset_sel][x]}",
        default=default_train_tickers,
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

    # Keep a per-run summary we can export
    if "train_results" not in st.session_state:
        st.session_state.train_results = []

    if train_btn:
        # reset summary for this run
        st.session_state.train_results = []
        if not tickers:
            st.warning("Enter tickers.")
        else:
            # UI elements for progress
            progress_bar = st.progress(0)
            status = st.empty()
            logs_box = st.empty()
            fold_timeline_slot = st.empty()
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
                    elif t == "cv_results":
                        mean_score = msg.get("mean_test_score")
                        std_score = msg.get("std_test_score")
                        split_scores = msg.get("split_test_scores", [])
                        n = msg.get("n_splits", 0)
                        fold_metrics = msg.get("fold_metrics", [])
                        fold_ranges = msg.get("fold_ranges", [])
                        # Format scores
                        mean_txt = f"mean={mean_score:.4f}" if mean_score is not None else ""
                        std_txt = f"std={std_score:.4f}" if std_score is not None else ""
                        splits_txt = ", ".join([f"fold{i+1}={s:.4f}" for i, s in enumerate(split_scores)])
                        # Format detailed per-fold metrics if present
                        detailed_txt = ""
                        if fold_metrics:
                            detailed_lines = []
                            for i, m in enumerate(fold_metrics):
                                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in m.items()])
                                detailed_lines.append(f"Fold {i+1}: {metrics_str}")
                            detailed_txt = "\n" + "\n".join(detailed_lines)
                        status_text = f"CV Results: {mean_txt}, {std_txt}, {splits_txt}{detailed_txt}"
                        status.markdown(status_text.replace("\n", "  "))
                        logs.insert(0, f"[cv_results] {status_text}")
                        logs_box.text_area("Training logs (most recent first)", value="\n".join(logs), height=260)
                        # Render fold ranges timeline if provided
                        if fold_ranges:
                            with fold_timeline_slot:
                                st.caption("Train/Test fold ranges")
                                render_fold_timeline(fold_ranges)
                    elif t == "done":
                        path = msg.get("path", "")
                        progress_bar.progress(100)
                        status.markdown(f"Saved model: {Path(path).name}")
                        logs.insert(0, f"[done] Saved model: {path}")
                        logs_box.text_area("Training logs (most recent first)", value="\n".join(logs), height=260)
                        # Append to run summary using metadata
                        try:
                            p = Path(path)
                            meta = load_model_metadata(p)
                            fname = p.name
                            # infer
                            model_type = (
                                "XGBoost" if fname.startswith("xgb_") else
                                "ANN" if fname.startswith("ann_") else
                                "Naive Bayes" if fname.startswith("nb_") else
                                "SVM" if fname.startswith("svm_") else "?"
                            )
                            task = "Classification" if "_class_" in fname else ("Regression" if "_reg_" in fname else "?")
                            # parse ticker and horizon from name like xgb_class_AAPL_h5.joblib
                            ticker_part = fname.split("_")[-2] if "_" in fname else "?"
                            import re
                            m = re.search(r"_h(\d+)", fname)
                            horizon_val = int(m.group(1)) if m else None
                            row = {
                                "file": fname,
                                "model": model_type,
                                "task": task,
                                "ticker": ticker_part,
                                "horizon": horizon_val,
                                "best_cv": meta.get("best_score") if meta else None,
                                "folds": meta.get("n_splits") if meta else None,
                                "train_rows": meta.get("train_rows") if meta else None,
                                "n_features": meta.get("n_features") if meta else None,
                                "train_time_sec": meta.get("duration_sec") if meta else None,
                                "date_start": meta.get("date_start") if meta else None,
                                "date_end": meta.get("date_end") if meta else None,
                                "path": str(p)
                            }
                            st.session_state.train_results.append(row)
                        except Exception:
                            pass
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
            # Show run summary with export option
            if st.session_state.train_results:
                st.markdown("---")
                st.subheader("Run summary")
                import pandas as _pd
                df_sum = _pd.DataFrame(st.session_state.train_results)
                st.dataframe(df_sum)
                colx, coly = st.columns([1,1])
                with colx:
                    if st.button("Export summary CSV", key="train_export_csv"):
                        rpt = ensure_reports_dir()
                        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
                        out_path = rpt / f"train_summary_{ts}.csv"
                        try:
                            df_sum.to_csv(out_path, index=False)
                            st.success(f"Saved: {out_path}")
                        except Exception as e:
                            st.error(f"Failed to save: {e}")
                with coly:
                    st.download_button("Download CSV", data=df_sum.to_csv(index=False), file_name="train_summary.csv", mime="text/csv", key="train_download_csv")
    # --- Holdout Evaluation Section ---
    st.markdown("---")
    st.header("Evaluate Model on Holdout Data (Unseen)")
    st.markdown("Test your trained models on unseen data to verify real-world performance.")

    import os
    from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

    # Model selection
    model_dir = str(MODELS_DIR)
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    if not model_files:
        st.info("No trained models found in /models. Train a model first.")
    else:
        model_choice = st.selectbox('Select a trained model:', model_files, key='eval_model_choice')

        # Data selection
        data_dir = str(BASE / 'data')
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not data_files:
            st.info("No CSV data files found in /data.")
        else:
            data_choice = st.selectbox('Select a dataset:', data_files, key='eval_data_choice')
            df = pd.read_csv(os.path.join(data_dir, data_choice))
            # Automatically add next_close if missing and Close exists
            if 'next_close' not in df.columns and 'Close' in df.columns:
                df['next_close'] = df['Close'].shift(-1)
                df = df.dropna(subset=['next_close'])
            task_type = st.radio('Task type:', ['Classification', 'Regression'], key='eval_task_type')
            test_size = st.slider('Holdout set size (percent)', 10, 50, 20, key='eval_test_size')
            train_df = df.iloc[:-int(len(df)*test_size/100)]
            holdout_df = df.iloc[-int(len(df)*test_size/100):]

            if st.button('Evaluate on Holdout', key='eval_run'):
                model_path = os.path.join(model_dir, model_choice)
                trained_model = load_model(model_path)
                # Apply feature engineering to holdout set
                from src.features import add_basic_features
                holdout_df_fe = add_basic_features(holdout_df.copy())
                # Force features to match those used during model training
                if hasattr(trained_model, 'feature_names_in_'):
                    features = list(trained_model.feature_names_in_)
                else:
                    # Fallback: use all columns except targets and date
                    features = [col for col in holdout_df_fe.columns if col not in ['label', 'next_close', 'date', 'Date']]
                X_holdout = holdout_df_fe[features]
                if task_type == 'Classification':
                    if 'label' not in holdout_df_fe.columns:
                        st.error("No 'label' column in holdout set after feature engineering.")
                    else:
                        y_holdout = holdout_df_fe['label']
                        preds = trained_model.predict(X_holdout)
                        if hasattr(trained_model, 'predict') and (preds.min() >= 0):
                            preds = preds - 1
                        report = classification_report(y_holdout, preds, output_dict=True)
                        st.subheader('Classification Report (Holdout Set)')
                        st.json(report)
                else:
                    if 'next_close' not in holdout_df_fe.columns:
                        st.error("No 'next_close' column in holdout set after feature engineering.")
                    else:
                        y_holdout = holdout_df_fe['next_close']
                        preds = trained_model.predict(X_holdout)
                        mae = mean_absolute_error(y_holdout, preds)
                        import numpy as np
                        rmse = np.sqrt(mean_squared_error(y_holdout, preds))
                        r2 = r2_score(y_holdout, preds)
                        st.subheader('Regression Metrics (Holdout Set)')
                        st.write(f'MAE: {mae:.4f}')
                        st.write(f'RMSE: {rmse:.4f}')
                        st.write(f'R2: {r2:.4f}')

    # --- Quick User Input Prediction ---
    st.markdown("---")
    st.header("Quick User Input Prediction")
    st.markdown("Run a fast prediction using a saved model on the latest rows for its ticker/horizon.")

    import re, os
    model_dir = str(MODELS_DIR)
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    if not model_files:
        st.info("No trained models found in /models. Train a model first.")
    else:
        user_model_choice = st.selectbox('Select a trained model:', model_files, key='user_pred_model_choice')
        # Parse info from filename like xgb_reg_AAPL_h1.joblib
        m = re.match(r"(xgb|ann|svm|nb)_(class|reg)_([^_]+)_h(\d+)\.joblib$", user_model_choice)
        info = None
        if m:
            fam, task, tk, h = m.groups()
            info = {"family": fam, "task": task, "ticker": tk, "horizon": int(h)}
            st.caption(f"Parsed — Ticker: {tk}, Horizon: {h} ({'days' if info['horizon']!=1 else 'day'}), Task: {task}")
        else:
            st.caption("Unrecognized filename pattern; attempting generic prediction.")

        # Date window and N rows
        up_start = st.text_input("Start date (YYYY-MM-DD)", "2018-01-01", key="user_pred_start")
        up_end = st.text_input("End date (YYYY-MM-DD or empty)", "", key="user_pred_end")
        N = st.slider("Rows to predict (latest N)", 1, 30, 5, key="user_pred_n")

        if st.button("Run user prediction", key="user_pred_run"):
            try:
                model_path = os.path.join(model_dir, user_model_choice)
                mdl = load_model(model_path)
                # Determine ticker/horizon
                ticker = info['ticker'] if info else None
                horizon = info['horizon'] if info else 1
                if not ticker:
                    st.error("Could not infer ticker from filename. Use standard naming: family_task_TICKER_hN.joblib")
                else:
                    df_raw = get_ticker_data(ticker, start=up_start or None, end=up_end or None)
                    if df_raw is None or df_raw.empty:
                        st.error("No data for selection.")
                    else:
                        df_raw = prepare_df_numeric(df_raw)
                        if info and info['task'] == 'class':
                            df = create_class_labels(df_raw, horizon=horizon)
                        else:
                            df = create_reg_target(df_raw, horizon=horizon)
                        df = add_basic_features(df)
                        if df is None or df.empty:
                            st.error("No valid data after feature engineering.")
                        else:
                            # Build features (numeric, exclude targets/labels/date)
                            drop_cols = {'Date','Ticker','next_close','ret_next','label'}
                            feats = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
                            if not feats:
                                st.error("No numeric features available for prediction.")
                            else:
                                sub = df.tail(N).copy()
                                if info and info['task'] == 'class':
                                    preds = predict_with_model(mdl, sub, feats, is_classifier=True)
                                    out = sub[['Date']].copy()
                                    out['True_label'] = sub['label'] if 'label' in sub.columns else np.nan
                                    out['Pred_label'] = preds
                                    st.subheader('Classification — latest predictions')
                                    st.dataframe(out)
                                else:
                                    preds = predict_with_model(mdl, sub, feats, is_classifier=False)
                                    # Align lengths if needed
                                    y_true = sub['next_close'] if 'next_close' in sub.columns else None
                                    if y_true is not None:
                                        k = min(len(preds), len(y_true))
                                        y_true = y_true.iloc[-k:]
                                        y_pred = np.array(preds)[-k:]
                                        out = pd.DataFrame({
                                            'Date': sub['Date'].iloc[-k:].values,
                                            'Actual_next_close': y_true.values,
                                            'Predicted_next_close': y_pred,
                                            'AbsError': np.abs(y_true.values - y_pred)
                                        })
                                    else:
                                        out = pd.DataFrame({
                                            'Date': sub['Date'].tail(len(preds)).values,
                                            'Predicted_next_close': np.array(preds)
                                        })
                                    st.subheader('Regression — latest predictions')
                                    st.dataframe(out)
            except Exception as e:
                st.error(f"User prediction failed: {e}")

    # --- One-click Batch Evaluation ---
    st.markdown("---")
    st.header("One-click Batch Evaluation")
    st.markdown("Evaluate many saved models on a chosen holdout window and export a consolidated CSV.")
    import re, os
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    if not model_files:
        st.info("No saved models found.")
    else:
        # Parse basic info from filenames like xgb_class_AAPL_h5.joblib
        def parse_model_name(fname: str):
            m = re.match(r"(xgb|ann|svm|nb)_(class|reg)_([^_]+)_h(\d+)\.joblib$", fname)
            if not m:
                return None
            family, task, ticker, h = m.groups()
            model_map = {"xgb":"XGBoost","ann":"ANN","svm":"SVM","nb":"Naive Bayes"}
            return {
                "file": fname,
                "family": family,
                "model": model_map.get(family, family),
                "task": "Classification" if task == "class" else "Regression",
                "ticker": ticker,
                "horizon": int(h)
            }

        parsed = [p for p in (parse_model_name(f) for f in model_files) if p]
        if not parsed:
            st.info("No parsable model filenames. Expected pattern: xgb_class_TICKER_hN.joblib")
        else:
            df_models = pd.DataFrame(parsed)
            task_sel = st.selectbox("Task", sorted(df_models['task'].unique().tolist()), key="batch_task")
            eligible = df_models[df_models['task'] == task_sel]
            tickers_opts = sorted(eligible['ticker'].unique().tolist())
            tickers_sel = st.multiselect("Tickers to evaluate", tickers_opts, default=tickers_opts[:5], key="batch_tickers")
            horizons_opts = sorted(eligible['horizon'].unique().tolist())
            horizon_sel = st.selectbox("Horizon", horizons_opts, index=0, key="batch_h")
            test_pct = st.slider('Holdout set size (percent)', 10, 50, 20, key='batch_test_size')
            eval_start = st.text_input("Start date (optional)", "2018-01-01", key="batch_start")
            eval_end = st.text_input("End date (optional)", "", key="batch_end")
            go = st.button("Run batch evaluation", key="batch_go")

            if go:
                rows = []
                for _, row in eligible[eligible['ticker'].isin(tickers_sel) & (eligible['horizon'] == horizon_sel)].iterrows():
                    fname = row['file']
                    ticker = row['ticker']
                    horizon = int(row['horizon'])
                    task = row['task']
                    model_type = row['model']
                    mpath = MODELS_DIR / fname
                    try:
                        df_raw = get_ticker_data(ticker, start=eval_start or None, end=eval_end or None)
                        if df_raw is None or df_raw.empty:
                            rows.append({"file": fname, "ticker": ticker, "result": "no-data"})
                            continue
                        df_raw = prepare_df_numeric(df_raw)
                        if task == "Classification":
                            df = create_class_labels(df_raw, horizon=horizon)
                            df = add_basic_features(df)
                            if df is None or df.empty or 'label' not in df.columns:
                                rows.append({"file": fname, "ticker": ticker, "result": "no-labels"})
                                continue
                            # split
                            k = max(1, int(len(df) * test_pct / 100))
                            train_df = df.iloc[:-k]
                            hold_df = df.iloc[-k:]
                            features = [c for c in df.columns if c not in {'Date','next_close','ret_next','label'} and np.issubdtype(df[c].dtype, np.number)]
                            model = load_model(mpath)
                            Xh = hold_df[features]
                            yh = hold_df['label']
                            preds = model.predict(Xh)
                            # If encoded as 0/1/2, remap to -1/0/1
                            try:
                                if preds.min() >= 0:
                                    preds = preds - 1
                            except Exception:
                                pass
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                            acc = accuracy_score(yh, preds)
                            prec = precision_score(yh, preds, average='weighted', zero_division=0)
                            rec = recall_score(yh, preds, average='weighted', zero_division=0)
                            f1 = f1_score(yh, preds, average='weighted', zero_division=0)
                            rows.append({
                                "file": fname, "model": model_type, "task": task, "ticker": ticker, "horizon": horizon,
                                "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
                                "rows_train": len(train_df), "rows_holdout": len(hold_df)
                            })
                        else:  # Regression
                            df = create_reg_target(df_raw, horizon=horizon)
                            df = add_basic_features(df)
                            if df is None or df.empty or 'next_close' not in df.columns:
                                rows.append({"file": fname, "ticker": ticker, "result": "no-target"})
                                continue
                            k = max(1, int(len(df) * test_pct / 100))
                            train_df = df.iloc[:-k]
                            hold_df = df.iloc[-k:]
                            features = [c for c in df.columns if c not in {'Date','next_close'} and np.issubdtype(df[c].dtype, np.number)]
                            model = load_model(mpath)
                            Xh = hold_df[features]
                            yh = hold_df['next_close']
                            preds = model.predict(Xh)
                            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                            mae = mean_absolute_error(yh, preds)
                            rmse = float(np.sqrt(mean_squared_error(yh, preds)))
                            r2 = r2_score(yh, preds)
                            rows.append({
                                "file": fname, "model": model_type, "task": task, "ticker": ticker, "horizon": horizon,
                                "MAE": mae, "RMSE": rmse, "R2": r2,
                                "rows_train": len(train_df), "rows_holdout": len(hold_df)
                            })
                    except Exception as e:
                        rows.append({"file": fname, "ticker": ticker, "result": f"error: {e}"})

                if rows:
                    df_out = pd.DataFrame(rows)
                    st.subheader("Batch results")
                    st.dataframe(df_out)
                    c1, c2 = st.columns([1,1])
                    with c1:
                        if st.button("Export to reports/", key="batch_export"):
                            try:
                                rpt = ensure_reports_dir()
                                ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
                                outp = rpt / f"batch_eval_{task_sel.lower()}_{ts}.csv"
                                df_out.to_csv(outp, index=False)
                                st.success(f"Saved: {outp}")
                            except Exception as e:
                                st.error(f"Save failed: {e}")
                    with c2:
                        st.download_button("Download CSV", data=df_out.to_csv(index=False), file_name="batch_eval.csv", mime="text/csv", key="batch_download")

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
    # Default to SVM on Assets page
    assets_model_default_index = model_options.index("SVM") if "SVM" in model_options else 0
    assets_model_type = st.selectbox(
        "Model",
        model_options,
        index=assets_model_default_index,
        key="assets_model_type"
    )
    st.markdown(f"<span style='font-size:0.98rem;color:#888'>{model_descriptions[assets_model_type]}</span>", unsafe_allow_html=True)

    # Default asset view to Stocks
    asset_choice = st.selectbox("Which assets to show", ("Stocks","Metals","Cryptocurrencies","All"), index=0, key="assets_choice")

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
# COMPARE (multi-ticker, multi-model classification)
# ---------------------------
if PAGE == "Compare":
    st.title("Compare Models — Classification")
    st.markdown("Pick tickers, horizon and models. We compute Accuracy, Precision, Recall, and F1, plus confusion matrices per ticker.")

    asset_map = build_asset_map_from_combined()
    # Inputs
    asset_type_list = list(asset_map.keys())
    asset_default_index = asset_type_list.index("Stocks") if "Stocks" in asset_type_list else 0
    asset_type = st.selectbox("Asset type", asset_type_list, index=asset_default_index, key="cmp_asset_type")

    available = asset_map[asset_type]
    preferred_stock_defaults = ["BAC", "NKE", "TSLA", "CRM", "INTC"]
    default_tickers = [t for t in preferred_stock_defaults if t in available.keys()] if asset_type == "Stocks" else list(available.keys())[:2]
    # Multiselect with Select All / Clear buttons
    # Initialize and control selection via session_state BEFORE rendering widget
    if "cmp_tickers" not in st.session_state:
        st.session_state.cmp_tickers = default_tickers
    c_sel_all, c_clear = st.columns([1,1])
    with c_sel_all:
        if st.button("Select All", key="cmp_sel_all"):
            st.session_state.cmp_tickers = list(available.keys())
            _safe_rerun()
    with c_clear:
        if st.button("Clear", key="cmp_sel_clear"):
            st.session_state.cmp_tickers = []
            _safe_rerun()
    tickers = st.multiselect(
        "Choose tickers",
        options=list(available.keys()),
        format_func=lambda x: f"{x} — {available[x]}",
        default=st.session_state.cmp_tickers,
        key="cmp_tickers"
    )

    horizon_map = {"1 day": 1, "1 week (~5 trading days)": 5, "1 month (~21 trading days)": 21}
    horizon_choice = st.selectbox("Prediction horizon", list(horizon_map.keys()), index=0, key="cmp_horizon")
    horizon = horizon_map[horizon_choice]

    model_choices = ["XGBoost", "ANN", "SVM", "Naive Bayes"]
    default_models = ["SVM", "XGBoost"]
    models_sel = st.multiselect("Models to compare", model_choices, default=default_models, key="cmp_models")

    start = st.text_input("Start date (YYYY-MM-DD)", "2018-01-01", key="cmp_start")
    end = st.text_input("End date (YYYY-MM-DD or empty)", "", key="cmp_end")
    btn = st.button("Run comparison", key="cmp_run")

    if btn:
        if not tickers:
            st.warning("Select at least one ticker.")
        elif not models_sel:
            st.warning("Select at least one model.")
        else:
            import pandas as pd
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            import plotly.figure_factory as ff

            rows = []
            for tk in tickers:
                # Load data
                df_raw = get_ticker_data(tk, start=start, end=end if end else None)
                if df_raw is None or df_raw.empty:
                    st.error(f"No data for {tk}")
                    continue
                df_raw = prepare_df_numeric(df_raw)
                df = create_class_labels(df_raw, horizon=horizon)
                df = add_basic_features(df)
                if df is None or df.empty or 'label' not in df.columns:
                    st.error(f"No valid classification data for {tk}")
                    continue
                features = [c for c in df.columns if c not in {'Date','next_close','ret_next','label'} and np.issubdtype(df[c].dtype, np.number)]
                y_true = df['label']

                # Prepare confusion matrix chart data container
                confmats = {}

                for mtype in models_sel:
                    if mtype == "XGBoost":
                        prefix = "xgb_class_"
                    elif mtype == "ANN":
                        prefix = "ann_class_"
                    elif mtype == "SVM":
                        prefix = "svm_class_"
                    elif mtype == "Naive Bayes":
                        prefix = "nb_class_"
                    else:
                        continue
                    model_path = safe_latest_model(prefix, tk, horizon=horizon)
                    if not model_path:
                        rows.append({"Ticker": tk, "Model": mtype, "Accuracy": None, "Precision": None, "Recall": None, "F1": None, "Info": "No model file"})
                        continue
                    try:
                        model = load_model(model_path)
                        preds = predict_with_model(model, df, features, is_classifier=True)
                        acc = accuracy_score(y_true, preds)
                        prec = precision_score(y_true, preds, average='weighted', zero_division=0)
                        rec = recall_score(y_true, preds, average='weighted', zero_division=0)
                        f1 = f1_score(y_true, preds, average='weighted', zero_division=0)
                        rows.append({"Ticker": tk, "Model": mtype, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "Info": ""})

                        # Confusion matrix per ticker per model
                        labels = sorted(list(set(y_true) | set(preds)))
                        cm = confusion_matrix(y_true, preds, labels=labels)
                        confmats[mtype] = (labels, cm)
                    except Exception as e:
                        rows.append({"Ticker": tk, "Model": mtype, "Accuracy": None, "Precision": None, "Recall": None, "F1": None, "Info": f"Error: {str(e)}"})

                # Render confusion matrices for this ticker
                if confmats:
                    st.markdown(f"### {tk} — Confusion Matrices")
                    cols = st.columns(max(1, min(3, len(confmats))))
                    i = 0
                    for mtype, (labels, cm) in confmats.items():
                        z = cm.astype(int)
                        z_text = [[str(v) for v in row] for row in z]
                        fig = ff.create_annotated_heatmap(z, x=[str(l) for l in labels], y=[str(l) for l in labels], colorscale='Blues', showscale=True, annotation_text=z_text)
                        fig.update_layout(title=f"{mtype}", xaxis_title="Predicted", yaxis_title="Actual")
                        with cols[i % len(cols)]:
                            st.plotly_chart(fig, use_container_width=True)
                        i += 1

            # Results table and evaluation
            st.markdown("---")
            st.subheader("Metrics Table")
            if rows:
                dfm = pd.DataFrame(rows)
                # Sort by Ticker then F1 desc
                with pd.option_context('display.precision', 4):
                    st.dataframe(dfm.sort_values(["Ticker", "F1"], ascending=[True, False]).set_index(["Ticker","Model"]))
                # Aggregate per model across tickers
                st.markdown("#### Average across selected tickers")
                agg = (
                    dfm.dropna(subset=["Accuracy"]) 
                    .groupby("Model")[ ["Accuracy", "Precision", "Recall", "F1"] ]
                    .mean()
                    .sort_values("F1", ascending=False)
                )
                st.dataframe(agg)
                # Evaluate overall best model: mean F1 (performance) and stability (std of F1 across tickers)
                st.markdown("#### Evaluation: Best Overall Model")
                perf = dfm.dropna(subset=["F1"]).groupby("Model")["F1"].agg(["mean","std"]).rename(columns={"mean":"F1_mean","std":"F1_std"})
                # Stability-first selection: lowest std (tie-breaker highest mean)
                perf = perf.sort_values(["F1_std", "F1_mean"], ascending=[True, False])
                best_model = perf.index[0] if len(perf) else None
                if best_model:
                    bm = perf.loc[best_model]
                    st.markdown(
                        f"""
                        <div style='border:1px solid #cfe8cf;background:#f2fbf2;padding:10px 14px;border-radius:8px;'>
                          <b>Chosen model:</b> <span style='color:#1e7e34'>{best_model}</span>
                          &nbsp;|&nbsp; F1 mean = <b>{bm['F1_mean']:.4f}</b>
                          &nbsp;|&nbsp; F1 std = <b>{(bm['F1_std'] if bm['F1_std']==bm['F1_std'] else 0):.4f}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # Show calculation table with std and highlight selected
                    def _hl(row):
                        color = '#fff3cd' if row.name == best_model else ''
                        return [f'background-color: {color}'] * len(row)
                    st.markdown("Calculation table (mean ± std)")
                    st.dataframe(
                        perf.rename(columns={"F1_mean":"F1_mean","F1_std":"F1_std"})
                            .fillna(0)
                            .style.format({"F1_mean":"{:.4f}", "F1_std":"{:.4f}"})
                            .apply(_hl, axis=1)
                    )
                else:
                    st.write("Chosen model:", "—")

                # Visualization: bar of F1_mean and error bars for stability; box plots per model
                try:
                    import plotly.graph_objects as go
                    fig1 = go.Figure()
                    x = perf.index.tolist()
                    y = perf["F1_mean"].tolist()
                    err = perf["F1_std"].fillna(0).tolist()
                    colors = ["#2ca02c" if m==best_model else "#1f77b4" for m in x]
                    fig1.add_trace(go.Bar(x=x, y=y, error_y=dict(type='data', array=err, visible=True), marker_color=colors))
                    fig1.update_layout(title="Overall Performance (F1 mean) with Stability (std)", yaxis_title="F1 (weighted)", xaxis_title="Model", height=380)
                    st.plotly_chart(fig1, use_container_width=True)

                    # Box plot of F1 per model across tickers
                    df_box = dfm.dropna(subset=["F1"]).copy()
                    import plotly.express as px
                    fig2 = px.box(df_box, x="Model", y="F1", points="all", color="Model")
                    fig2.update_layout(title="Distribution of F1 by Model (stability across tickers)", height=380, showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.info(f"Plotting issue: {e}")
                # Styled tables highlighting best model row
                try:
                    def _hl_row(row):
                        style = [""] * len(row)
                        if row.name == best_model:
                            style = ["background-color: #e7f7e7; font-weight:700" for _ in row]
                        return style
                    st.markdown("##### F1 mean/std (highlighted)")
                    st.dataframe(perf.style.apply(_hl_row, axis=1))
                    st.markdown("##### Aggregated metrics (highlighted by best F1)")
                    st.dataframe(agg.style.apply(lambda r: ["background-color: #e7f7e7; font-weight:700" if r.name==best_model else "" for _ in r], axis=1))
                except Exception:
                    pass
                # Export options
                col_a, col_b = st.columns([1,1])
                with col_a:
                    if st.button("Export results to reports/", key="cmp_export"):
                        try:
                            rpt = ensure_reports_dir()
                            ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
                            out1 = rpt / f"compare_rows_{ts}.csv"
                            out2 = rpt / f"compare_agg_{ts}.csv"
                            dfm.to_csv(out1, index=False)
                            agg.to_csv(out2)
                            st.success(f"Saved: {out1.name}, {out2.name} in {rpt}")
                        except Exception as e:
                            st.error(f"Save failed: {e}")
                with col_b:
                    st.download_button("Download metrics CSV", data=dfm.to_csv(index=False), file_name="compare_metrics.csv", mime="text/csv", key="cmp_download")
            else:
                st.info("No results generated.")

# ---------------------------
# SUMMARY PAGE (unique key)
# ---------------------------


if PAGE == "Summary":
    st.title("Model Comparison Summary")
    st.markdown("Select a stock to compare predictions from all models.")

    asset_map = build_asset_map_from_combined()
    asset_type = st.selectbox("Asset type", list(asset_map.keys()), key="summary_asset_type")
    available = asset_map[asset_type]
    # Switch to multiselect with Select All for broader comparison; default to first 1-3
    tickers_all = list(available.keys())
    default_tks = tickers_all[:3]
    if "summary_tickers" not in st.session_state:
        st.session_state.summary_tickers = default_tks
    # Flags to safely update selection before widget instantiation
    if "sum_select_all_flag" not in st.session_state:
        st.session_state.sum_select_all_flag = False
    if "sum_clear_flag" not in st.session_state:
        st.session_state.sum_clear_flag = False
    s_all, s_clear = st.columns([1,1])
    with s_all:
        st.button(
            "Select All",
            key="sum_sel_all",
            on_click=lambda: st.session_state.update({"sum_select_all_flag": True})
        )
    with s_clear:
        st.button(
            "Clear",
            key="sum_sel_clear",
            on_click=lambda: st.session_state.update({"sum_clear_flag": True})
        )
    # Apply pending selection changes BEFORE creating the multiselect widget
    if st.session_state.sum_select_all_flag:
        st.session_state.summary_tickers = tickers_all
        st.session_state.sum_select_all_flag = False
    if st.session_state.sum_clear_flag:
        st.session_state.summary_tickers = []
        st.session_state.sum_clear_flag = False
    tickers = st.multiselect("Choose tickers", tickers_all, default=st.session_state.summary_tickers, format_func=lambda x: f"{x} — {available[x]}", key="summary_tickers")
    horizon_map = {"1 day": 1, "1 week (~5 trading days)": 5, "1 month (~21 trading days)": 21}
    horizon_choice = st.selectbox("Prediction horizon", list(horizon_map.keys()), index=0, key="summary_horizon")
    horizon = horizon_map[horizon_choice]
    mode = st.selectbox("Task", ["Classification (direction)", "Regression (price)"], key="summary_mode")
    start = st.text_input("Start date (YYYY-MM-DD)", "2018-01-01", key="summary_start")
    end = st.text_input("End date (YYYY-MM-DD or empty)", "", key="summary_end")

    # Guard
    if not tickers:
        st.info("Select at least one ticker.")
    else:
        # For each selected ticker, compute metrics per model; then aggregate across tickers for evaluation
        model_types = ["XGBoost", "ANN", "SVM", "Naive Bayes"]
        all_rows = []
        # For regression plotting: store per-ticker predictions per model
        per_ticker_preds = {}
        debug_msgs = []
        for ticker in tickers:
            df_raw = get_ticker_data(ticker, start=start, end=end if end else None)
            if df_raw is None or df_raw.empty:
                debug_msgs.append(f"[INFO] No data for {ticker}")
                continue
            df_raw = prepare_df_numeric(df_raw)
            if mode.startswith("Class"):
                df = create_class_labels(df_raw, horizon=horizon)
            else:
                df = create_reg_target(df_raw, horizon=horizon)
            df = add_basic_features(df)
            if df is None or df.empty:
                debug_msgs.append(f"[INFO] No valid data for {ticker}")
                continue
            features = [c for c in df.columns if c not in {'Date','next_close','ret_next','label'} and np.issubdtype(df[c].dtype, np.number)]
            for mtype in model_types:
                if mtype == "XGBoost":
                    prefix = "xgb_class_" if mode.startswith("Class") else "xgb_reg_"
                elif mtype == "ANN":
                    prefix = "ann_class_" if mode.startswith("Class") else "ann_reg_"
                elif mtype == "SVM":
                    prefix = "svm_class_" if mode.startswith("Class") else "svm_reg_"
                elif mtype == "Naive Bayes":
                    if not mode.startswith("Class"):
                        debug_msgs.append("[INFO] Naive Bayes: classification only.")
                        continue
                    prefix = "nb_class_"
                else:
                    continue
                model_path = safe_latest_model(prefix, ticker, horizon=horizon)
                if not model_path:
                    continue
                try:
                    model = load_model(model_path)
                    preds = predict_with_model(model, df, features, is_classifier=mode.startswith("Class"))
                    if mode.startswith("Class"):
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                        if 'label' not in df.columns:
                            continue
                        y_true = df['label']
                        acc = accuracy_score(y_true, preds)
                        prec = precision_score(y_true, preds, average='weighted', zero_division=0)
                        rec = recall_score(y_true, preds, average='weighted', zero_division=0)
                        f1 = f1_score(y_true, preds, average='weighted', zero_division=0)
                        all_rows.append({"Ticker": ticker, "Model": mtype, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
                    else:
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                        if 'next_close' not in df.columns:
                            continue
                        y_true = df['next_close']
                        # align preds length
                        import numpy as _np
                        y_true = y_true.iloc[-len(preds):]
                        preds_s = pd.Series(preds, index=y_true.index)
                        mae = mean_absolute_error(y_true, preds_s)
                        rmse = float(_np.sqrt(mean_squared_error(y_true, preds_s)))
                        r2 = r2_score(y_true, preds_s)
                        all_rows.append({"Ticker": ticker, "Model": mtype, "MAE": mae, "RMSE": rmse, "R2": r2})
                        # Save for plotting
                        if ticker not in per_ticker_preds:
                            per_ticker_preds[ticker] = {"dates": y_true.index.to_list(), "actual": y_true.to_list(), "models": {}}
                        per_ticker_preds[ticker]["models"][mtype] = preds_s.to_list()
                except Exception:
                    continue

        if not all_rows:
            st.info("No results available for the selected configuration.")
        else:
            dfm = pd.DataFrame(all_rows)
            if mode.startswith("Class"):
                st.subheader("Classification Metrics — All Selected Tickers")
                st.dataframe(dfm.set_index(["Ticker","Model"]))
                # Evaluation
                st.markdown("#### Evaluation: Best Overall Model")
                perf = dfm.groupby("Model")["F1"].agg(["mean","std"]).rename(columns={"mean":"F1_mean","std":"F1_std"}).sort_values(["F1_std","F1_mean"], ascending=[True, False])
                best_model = perf.index[0]
                # Highlight card
                bm = perf.loc[best_model]
                st.markdown(
                    f"""
                    <div style='border:1px solid #cfe8cf;background:#f2fbf2;padding:10px 14px;border-radius:8px;'>
                        <b>Chosen model:</b> <span style='color:#1e7e34'>{best_model}</span>
                        &nbsp;|&nbsp; F1 mean = <b>{bm['F1_mean']:.4f}</b>
                        &nbsp;|&nbsp; F1 std = <b>{(bm['F1_std'] if bm['F1_std']==bm['F1_std'] else 0):.4f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Show calculation table with std and highlight selected
                def _hl2(row):
                    color = '#fff3cd' if row.name == best_model else ''
                    return [f'background-color: {color}'] * len(row)
                st.markdown("Calculation table (mean ± std)")
                st.dataframe(
                    perf.fillna(0)
                        .style.format({"F1_mean":"{:.4f}", "F1_std":"{:.4f}"})
                        .apply(_hl2, axis=1)
                )
                # Charts
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    fig1 = go.Figure()
                    fig1.add_trace(go.Bar(x=perf.index, y=perf['F1_mean'], error_y=dict(type='data', array=perf['F1_std'].fillna(0)), marker_color=["#2ca02c" if m==best_model else "#1f77b4" for m in perf.index]))
                    fig1.update_layout(title="Overall Performance (F1 mean) with Stability (std)", yaxis_title="F1 (weighted)")
                    st.plotly_chart(fig1, use_container_width=True)
                    fig2 = px.box(dfm, x="Model", y="F1", color="Model")
                    fig2.update_layout(title="F1 Distribution by Model across Selected Tickers", showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.info(f"Plotting issue: {e}")
            else:
                st.subheader("Regression Metrics — All Selected Tickers")
                st.dataframe(dfm.set_index(["Ticker","Model"]))
                # Stability-first for regression: pick lowest RMSE std (tie: lowest RMSE mean, then highest R2 mean)
                perf = (
                    dfm.groupby("Model")
                    .agg(
                        MAE_mean=("MAE","mean"), RMSE_mean=("RMSE","mean"), R2_mean=("R2","mean"),
                        MAE_std=("MAE","std"), RMSE_std=("RMSE","std"), R2_std=("R2","std"),
                    )
                    .fillna(0)
                )
                perf = perf.sort_values(["RMSE_std","RMSE_mean","R2_mean"], ascending=[True, True, False])
                best_model = perf.index[0]
                st.markdown("#### Evaluation: Best Overall Model")
                st.markdown(
                    f"""
                    <div style='border:1px solid #cfe8cf;background:#f2fbf2;padding:10px 14px;border-radius:8px;'>
                        <b>Chosen model:</b> <span style='color:#1e7e34'>{best_model}</span>
                        &nbsp;|&nbsp; Mean MAE = <b>{perf.loc[best_model,'MAE_mean']:.4f}</b>
                        &nbsp;|&nbsp; Mean RMSE = <b>{perf.loc[best_model,'RMSE_mean']:.4f}</b>
                        &nbsp;|&nbsp; Mean R2 = <b>{perf.loc[best_model,'R2_mean']:.4f}</b>
                        &nbsp;|&nbsp; RMSE std = <b>{perf.loc[best_model,'RMSE_std']:.4f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Show calculation table with stds and highlight selected
                def _hl3(row):
                    color = '#fff3cd' if row.name == best_model else ''
                    return [f'background-color: {color}'] * len(row)
                st.markdown("Calculation table (means ± stds)")
                st.dataframe(
                    perf
                        .style.format({
                            "MAE_mean":"{:.4f}", "RMSE_mean":"{:.4f}", "R2_mean":"{:.4f}",
                            "MAE_std":"{:.4f}", "RMSE_std":"{:.4f}", "R2_std":"{:.4f}"
                        })
                        .apply(_hl3, axis=1)
                )
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='MAE (lower better)', x=perf.index, y=perf['MAE_mean']))
                    fig.add_trace(go.Bar(name='RMSE (lower better)', x=perf.index, y=perf['RMSE_mean']))
                    fig.add_trace(go.Bar(name='R2 (higher better)', x=perf.index, y=perf['R2_mean']))
                    fig.update_layout(barmode='group', title='Overall Regression Performance (means)')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Plotting issue: {e}")

                # Per-ticker overlay charts: Actual vs Predicted (per model)
                if per_ticker_preds:
                    st.markdown("#### Per‑ticker Prediction vs Actual")
                    for tk in tickers:
                        if tk not in per_ticker_preds:
                            continue
                        try:
                            import plotly.graph_objects as go
                            payload = per_ticker_preds[tk]
                            dates = payload["dates"]
                            figp = go.Figure()
                            figp.add_trace(go.Scatter(x=dates, y=payload["actual"], mode='lines', name='Actual', line=dict(color='black')))
                            color_map = {'XGBoost':'#1f77b4','ANN':'#2ca02c','SVM':'#d62728','Naive Bayes':'#9467bd'}
                            for mtype, yhat in payload["models"].items():
                                figp.add_trace(go.Scatter(x=dates, y=yhat, mode='lines', name=mtype, line=dict(color=color_map.get(mtype, None), dash='dash')))
                            figp.update_layout(title=f"{tk} — Actual vs Predicted Next Close", xaxis_title="Date", yaxis_title="Price", height=420)
                            with st.expander(f"{tk} chart", expanded=False):
                                st.plotly_chart(figp, use_container_width=True)
                        except Exception as e:
                            st.info(f"Chart unavailable for {tk}: {e}")
# ---------------------------
# ABOUT
# ---------------------------
if PAGE == "About":
    st.title("About")
    st.markdown("This app is a demo for coursework. Predictions are illustrative, not financial advice.")
    st.markdown("- Data source: Yahoo Finance (yfinance)")
    st.markdown("- Models: XGBoost (trained with TimeSeriesSplit)")

# end
