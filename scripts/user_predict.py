import sys
from pathlib import Path
import pandas as pd
import numpy as np

# App modules
from src.labeling import create_reg_target
from src.features import add_basic_features
from src.model import load_model, predict_with_model

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "historical_all.csv"
MODELS = BASE / "models"

# --- User inputs ---
TICKER = "AAPL"
HORIZON = 1  # days ahead
START = "2018-01-01"
END = None  # e.g., "2024-12-31" or None
MODEL_KIND = "xgb_reg"  # try xgb_reg; fallback could be svm_reg if needed

# --- Load data ---
if not DATA.exists():
    print(f"ERR: data file not found: {DATA}")
    sys.exit(1)

df = pd.read_csv(DATA, parse_dates=["Date"]) 
# filter ticker and date window
mask = (df["Ticker"] == TICKER)
if START:
    mask &= (df["Date"] >= pd.to_datetime(START))
if END:
    mask &= (df["Date"] <= pd.to_datetime(END))

df = df.loc[mask].copy()
if df.empty:
    print("ERR: no rows for selection")
    sys.exit(1)

# Basic cleaning: ensure numeric for common cols
num_cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = create_reg_target(df, horizon=HORIZON)
df = add_basic_features(df)

if "next_close" not in df.columns:
    print("ERR: missing next_close target after feature engineering")
    sys.exit(1)

# Feature set mirrors app logic (exclude target/labels/date)
features = [c for c in df.columns if c not in {"Date","next_close","ret_next","label","Ticker"} and np.issubdtype(df[c].dtype, np.number)]

# --- Load model ---
model_path = MODELS / f"{MODEL_KIND}_{TICKER}_h{HORIZON}.joblib"
if not model_path.exists():
    print(f"ERR: model not found: {model_path}")
    # Suggest alternatives present
    alts = sorted([p.name for p in MODELS.glob(f"*_{TICKER}_h{HORIZON}.joblib")])
    if alts:
        print("Available for this ticker/horizon:", ", ".join(alts))
    sys.exit(1)

model = load_model(model_path)

# --- Predict (use last N rows as the 'user input' horizon window) ---
N = 5
sub = df.tail(N).copy()
y_true = sub["next_close"]
y_pred = predict_with_model(model, sub, features, is_classifier=False)

# Align (in case pipeline trims)
if len(y_pred) != len(y_true):
    k = min(len(y_pred), len(y_true))
    y_true = y_true.iloc[-k:]
    y_pred = np.array(y_pred)[-k:]
else:
    y_pred = np.array(y_pred)

# Report
out = pd.DataFrame({
    "Date": sub["Date"].iloc[-len(y_true):].dt.strftime("%Y-%m-%d").values,
    "Actual_next_close": y_true.values,
    "Predicted_next_close": y_pred,
    "AbsError": np.abs(y_true.values - y_pred)
})

print("Model:", model_path.name)
print("Ticker:", TICKER, "Horizon:", HORIZON)
print(out.to_string(index=False))
