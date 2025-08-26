from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
def train_nb_classifier(df, features, label_col='label', save_name='nb_class.joblib', n_splits=5, progress_callback=None):
    """
    Train Naive Bayes classifier with TimeSeriesSplit.
    """
    df = df.copy().reset_index(drop=True)
    X = df[features]
    y = df[label_col]
    y_m = (y + 1).astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    last_model = None
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if progress_callback:
            progress_callback({"type": "fold_start", "fold": i, "n_splits": n_splits})
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_m.iloc[train_idx], y_m.iloc[test_idx]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('nb', GaussianNB())
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        preds_orig = preds - 1
        y_test_orig = y_test - 1
        metrics = _compute_class_metrics(y_test_orig, preds_orig)
        if progress_callback:
            progress_callback({"type": "fold", "fold": i, "n_splits": n_splits, "metrics": metrics})
        last_model = pipe
    out = MODELS_DIR / save_name
    joblib.dump(last_model, out)
    if progress_callback:
        progress_callback({"type": "done", "path": str(out)})
    return out

def train_svm_classifier(df, features, label_col='label', save_name='svm_class.joblib', n_splits=5, progress_callback=None):
    """
    Train SVM classifier with TimeSeriesSplit.
    """
    df = df.copy().reset_index(drop=True)
    X = df[features]
    y = df[label_col]
    y_m = (y + 1).astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    last_model = None
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if progress_callback:
            progress_callback({"type": "fold_start", "fold": i, "n_splits": n_splits})
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_m.iloc[train_idx], y_m.iloc[test_idx]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        preds_orig = preds - 1
        y_test_orig = y_test - 1
        metrics = _compute_class_metrics(y_test_orig, preds_orig)
        if progress_callback:
            progress_callback({"type": "fold", "fold": i, "n_splits": n_splits, "metrics": metrics})
        last_model = pipe
    out = MODELS_DIR / save_name
    joblib.dump(last_model, out)
    if progress_callback:
        progress_callback({"type": "done", "path": str(out)})
    return out

def train_svm_regressor(df, features, target_col='next_close', save_name='svm_reg.joblib', n_splits=5, progress_callback=None):
    """
    Train SVM regressor with TimeSeriesSplit.
    """
    df = df.copy().reset_index(drop=True)
    X = df[features]
    y = df[target_col]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    last_model = None
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if progress_callback:
            progress_callback({"type": "fold_start", "fold": i, "n_splits": n_splits})
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds))
        }
        if progress_callback:
            progress_callback({"type": "fold", "fold": i, "n_splits": n_splits, "metrics": metrics})
        last_model = pipe
    out = MODELS_DIR / save_name
    joblib.dump(last_model, out)
    if progress_callback:
        progress_callback({"type": "done", "path": str(out)})
    return out
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def _compute_class_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }

def train_xgb_classifier(df, features, label_col='label', save_name='xgb_class.joblib', n_splits=5, progress_callback=None):
    """
    Train classifier with TimeSeriesSplit. If progress_callback is provided, it will be called with
    a dict argument like {'type':'fold', 'fold':i, 'n_splits':n_splits, 'metrics': metrics_dict}
    and with {'type':'done','path': out}.
    """
    df = df.copy().reset_index(drop=True)
    X = df[features]
    y = df[label_col]
    # map -1,0,1 -> 0,1,2 for XGB labels (this preserves your previous behaviour)
    y_m = (y + 1).astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    last_model = None
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if progress_callback:
            progress_callback({"type": "fold_start", "fold": i, "n_splits": n_splits})
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_m.iloc[train_idx], y_m.iloc[test_idx]
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100))])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        # convert back to -1/0/1 for reporting convenience
        preds_orig = preds - 1
        y_test_orig = y_test - 1
        metrics = _compute_class_metrics(y_test_orig, preds_orig)
        if progress_callback:
            progress_callback({"type": "fold", "fold": i, "n_splits": n_splits, "metrics": metrics})
        last_model = pipe

    out = MODELS_DIR / save_name
    joblib.dump(last_model, out)
    if progress_callback:
        progress_callback({"type": "done", "path": str(out)})
    return out

def train_ann_classifier(df, features, label_col='label', save_name='ann_class.joblib', n_splits=5, progress_callback=None):
    """
    Train ANN (MLP) classifier with TimeSeriesSplit. Same interface as XGB.
    """
    df = df.copy().reset_index(drop=True)
    X = df[features]
    y = df[label_col]
    y_m = (y + 1).astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    last_model = None
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if progress_callback:
            progress_callback({"type": "fold_start", "fold": i, "n_splits": n_splits})
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_m.iloc[train_idx], y_m.iloc[test_idx]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        preds_orig = preds - 1
        y_test_orig = y_test - 1
        metrics = _compute_class_metrics(y_test_orig, preds_orig)
        if progress_callback:
            progress_callback({"type": "fold", "fold": i, "n_splits": n_splits, "metrics": metrics})
        last_model = pipe
    out = MODELS_DIR / save_name
    joblib.dump(last_model, out)
    if progress_callback:
        progress_callback({"type": "done", "path": str(out)})
    return out

def train_xgb_regressor(df, features, target_col='next_close', save_name='xgb_reg.joblib', n_splits=5, progress_callback=None):
    """
    Train regressor with TimeSeriesSplit. progress_callback similar to classifier.
    """
    df = df.copy().reset_index(drop=True)
    X = df[features]
    y = df[target_col]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    last_model = None
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if progress_callback:
            progress_callback({"type": "fold_start", "fold": i, "n_splits": n_splits})
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipe = Pipeline([('scaler', StandardScaler()), ('xgb', XGBRegressor(n_estimators=150))])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds))
        }
        if progress_callback:
            progress_callback({"type": "fold", "fold": i, "n_splits": n_splits, "metrics": metrics})
        last_model = pipe

    out = MODELS_DIR / save_name
    joblib.dump(last_model, out)
    if progress_callback:
        progress_callback({"type": "done", "path": str(out)})
    return out

def train_ann_regressor(df, features, target_col='next_close', save_name='ann_reg.joblib', n_splits=5, progress_callback=None):
    """
    Train ANN (MLP) regressor with TimeSeriesSplit. Same interface as XGB.
    """
    df = df.copy().reset_index(drop=True)
    X = df[features]
    y = df[target_col]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    last_model = None
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if progress_callback:
            progress_callback({"type": "fold_start", "fold": i, "n_splits": n_splits})
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds))
        }
        if progress_callback:
            progress_callback({"type": "fold", "fold": i, "n_splits": n_splits, "metrics": metrics})
        last_model = pipe
    out = MODELS_DIR / save_name
    joblib.dump(last_model, out)
    if progress_callback:
        progress_callback({"type": "done", "path": str(out)})
    return out

def load_model(path):
    return joblib.load(path)

def predict_with_model(model, X_df, features, is_classifier=True):
    X = X_df[features]
    preds = model.predict(X)
    if is_classifier:
        # 0/1/2 -> -1/0/1 (app expects -1/0/1)
        return preds - 1
    return preds
