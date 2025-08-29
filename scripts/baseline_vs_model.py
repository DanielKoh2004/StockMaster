import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features import add_basic_features
from src.labeling import create_reg_target
from src.model import train_xgb_regressor

# Load data
file = "data/AAPL.csv"
df = pd.read_csv(file)

# Prepare data (use horizon=1 for next day prediction)
df = create_reg_target(df, horizon=1)
df = add_basic_features(df)

# Baseline: predict next_close as previous day's close
# Align so that y_true and y_pred are the same length
baseline_pred = df['Close'].shift(1)
y_true = df['next_close']

# Drop first row (where baseline_pred is NaN)
mask = ~baseline_pred.isna()
y_true = y_true[mask]
baseline_pred = baseline_pred[mask]

mae = mean_absolute_error(y_true, baseline_pred)
rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
r2 = r2_score(y_true, baseline_pred)

print("Baseline (Prev Close) Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# Inspect features and target
print("\nSample features and target:")
print(df.head(10)[['Close','next_close','returns','ma5','ma10','ma20','vol10','rsi14','lag_ret_1','lag_ret_2','lag_ret_3','lag_ret_5','ma5_ma10']])

# (Optional) You can add more features here, e.g.:
# df['vol_ma5'] = df['Volume'].rolling(5).mean()
# df['vol_change'] = df['Volume'].pct_change()
# ...

# You can also run your model here and compare metrics if desired.
