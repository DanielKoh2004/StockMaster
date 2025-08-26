import pandas as pd

def create_class_labels(df, horizon: int = 1, thr_up: float = 0.005, thr_down: float = -0.005):
    """
    Create classification labels for a horizon (shift by -horizon).
    Returns label values in -1/0/1 (Down/Stable/Up) to remain compatible with existing model pipeline.
    """
    df = df.copy().reset_index(drop=True)
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    df['next_close'] = df['Close'].shift(-horizon)
    df['ret_next'] = (df['next_close'] - df['Close']) / df['Close']
    def lab(x):
        if x > thr_up:
            return 1
        if x < thr_down:
            return -1
        return 0
    df['label'] = df['ret_next'].apply(lab)
    df = df.dropna().reset_index(drop=True)
    return df

def create_reg_target(df, horizon: int = 1):
    """
    Create regression target next_close at specified horizon (shift -horizon).
    """
    df = df.copy().reset_index(drop=True)
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    df['next_close'] = df['Close'].shift(-horizon)
    df = df.dropna().reset_index(drop=True)
    return df
