import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import ta  # pip install ta


# Load data and restrict to 2018-01-01 to 2018-05-30
file = "data/AAPL.csv"
df = pd.read_csv(file)
# Robust date parsing and filtering
df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
start_date = pd.Timestamp('2018-01-01', tz='UTC')
end_date = pd.Timestamp('2018-05-30', tz='UTC')
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)

# Feature engineering
# Add basic features
from src.features import add_basic_features
from src.labeling import create_reg_target

df = create_reg_target(df, horizon=1)
df = add_basic_features(df)

# Add more lags
for l in [7, 14, 21]:
    df[f'lag_ret_{l}'] = df['returns'].shift(l)

# Add volume-based features
for w in [5, 10, 20]:
    df[f'vol_ma{w}'] = df['Volume'].rolling(w).mean()
    df[f'vol_std{w}'] = df['Volume'].rolling(w).std()
df['vol_change'] = df['Volume'].pct_change()

# Add technical indicators using ta
# MACD
macd = ta.trend.MACD(df['Close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_diff'] = macd.macd_diff()
# Bollinger Bands
bb = ta.volatility.BollingerBands(df['Close'])
df['bb_high'] = bb.bollinger_hband()
df['bb_low'] = bb.bollinger_lband()
# Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
df['stoch_k'] = stoch.stoch()
df['stoch_d'] = stoch.stoch_signal()

# Drop rows with NaNs from feature creation
features = [c for c in df.columns if c not in {'Date','next_close'} and df[c].dtype != 'O']
df = df.dropna(subset=features + ['next_close']).reset_index(drop=True)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df['next_close']

# Baseline: previous close
baseline_pred = df['Close'].shift(1)
y_true = df['next_close']
mask = ~baseline_pred.isna()
y_true = y_true[mask]
baseline_pred = baseline_pred[mask]
print("Baseline (Prev Close) Metrics:")
print(f"MAE: {mean_absolute_error(y_true, baseline_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, baseline_pred)):.4f}")
print(f"R2: {r2_score(y_true, baseline_pred):.4f}")


# Model tuning with TimeSeriesSplit (5 folds) and GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 2]
}

tscv = TimeSeriesSplit(n_splits=5)
xgb = XGBRegressor(tree_method='hist', random_state=42)
gs = GridSearchCV(xgb, param_grid, cv=tscv, scoring='r2', n_jobs=1, verbose=1)
gs.fit(X, y)

print("\nBest XGBoost Params:")
print(gs.best_params_)
print(f"Best CV R2: {gs.best_score_:.4f}")

# Evaluate on all data
y_pred = gs.predict(X)
print("\nTuned XGBoost Metrics (all data):")
print(f"MAE: {mean_absolute_error(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
print(f"R2: {r2_score(y, y_pred):.4f}")

# Inspect features and target
print("\nSample features and target:")
print(df.head(10)[features + ['next_close']])
