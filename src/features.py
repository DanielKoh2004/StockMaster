import pandas as pd

def add_basic_features(df, lags=[1,2,3,5]):
    df = df.copy().reset_index(drop=True)
    df['returns'] = df['Close'].pct_change()
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma10'] = df['Close'].rolling(10).mean()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['vol10'] = df['Close'].rolling(10).std()
    # simple RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100 - (100 / (1 + rs))
    for l in lags:
        df[f'lag_ret_{l}'] = df['returns'].shift(l)
    # Add more lags
    for l in [7, 14, 21]:
        df[f'lag_ret_{l}'] = df['returns'].shift(l)
    # Add volume-based features
    for w in [5, 10, 20]:
        df[f'vol_ma{w}'] = df['Volume'].rolling(w).mean()
        df[f'vol_std{w}'] = df['Volume'].rolling(w).std()
    df['vol_change'] = df['Volume'].pct_change()
    # Add technical indicators using ta
    import ta
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['ma5_ma10'] = df['ma5'] - df['ma10']
    df = df.dropna().reset_index(drop=True)
    return df
