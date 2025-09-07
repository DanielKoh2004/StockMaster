# StockMaster

StockMaster is a modern, interactive Streamlit app for stock, crypto, and commodity price prediction and analysis. It uses machine learning models (XGBoost, SVM, ANN, and Naive Bayes) to forecast price direction and magnitude, and provides a visually rich dashboard for exploring predictions, historical data, and model performance.

## Features
- Dynamic Data: Local cached CSVs with friendly cleaning; logo fetchers for nicer cards.
- Predictions: Regression (next-close) and Classification (up/stable/down); multi‑model support (XGBoost, SVM, ANN; Naive Bayes for classification).
- Visual Dashboard: Home, Assets, Predictor, Compare, Train, Models, Summary, About; sticky navbar and branding.
- Stock/Crypto/Commodity: Works for stocks, cryptocurrencies, and futures/metals (with a few excluded tickers for clarity).
- Model Training: Train XGBoost/SVM/ANN/NB from the UI with TimeSeriesSplit cross‑validation.
- Model Cards: Best CV score, folds, training window, features, train time, and params (when available).
- Deeper Classification Analysis: Threshold slider, confusion matrix, macro ROC/PR AUCs, misclassification table.
- Compare Page: Multi‑ticker, multi‑model metrics with per‑ticker confusion matrices and export to CSV.
- Fold Timelines: Visualize train/test windows per fold during training.
- Data Health Panel: Quick snapshot of dataset rows, ticker count, and last date on Home.
- Exports: One‑click exports for training run summaries, compare results, and batch evaluation to the reports/ folder.

### Stability‑first model selection (how “best” is chosen)
- Classification: the most stable model is the one with the lowest F1 standard deviation across TimeSeriesSplit folds (F1_std). Ties are broken by higher F1_mean.
- Regression: the most stable model is the one with the lowest RMSE standard deviation (RMSE_std). Ties are broken by lower RMSE_mean, then higher R²_mean.
- Compare and Summary pages show transparent “calculation tables” with mean ± std and highlight the chosen row.

## Installation
1. Clone the repository or download the source code.
2. Install dependencies:
    - Windows PowerShell example:
       ```powershell
       python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
       ```
3. (Optional) Place your CSVs in the `data/` folder for offline use.

## Usage
Run the app from the project root:
```powershell
python -m streamlit run src/app.py
```

Tip: Models are read from `models/`. Train them on the Train page first if empty.

### Model file naming convention
Saved models follow a simple pattern so the app can auto‑detect the target asset and horizon:

```
<algo>_<task>_<TICKER>_h<H>
```

Examples:
- `svm_reg_BAC_h1.joblib` → SVM regression for ticker BAC, horizon 1 day
- `xgb_cls_AAPL_h1.joblib` → XGBoost classification for ticker AAPL, horizon 1 day

Sidecar metadata files (e.g., `*.meta.json`) store the CV fold scores, best params, and training window used.

## Folder Structure
```
StockMaster/
├── data/           # Single-ticker CSVs and historical data
├── models/         # Trained model files (.joblib) + sidecar *.meta.json
├── notebooks/      # (Optional) Jupyter notebooks
├── reports/        # (Optional) Reports/outputs
├── scripts/        # Data download/compile scripts
├── src/            # Main app and source code
│   └── app.py      # Streamlit app entry point
├── requirements.txt
├── README.md
└── .streamlit/     # (Optional) Streamlit config/theme
```

## Requirements
- Python 3.9+
- See `requirements.txt` for dependencies (streamlit, scikit‑learn, xgboost, plotly, pandas, yfinance, ta, joblib, pillow, requests)

## Notes & Tips
- Data Health panel on Home shows cached dataset status.
- Train page shows fold timelines and logs. After a run, export a CSV summary via the button.
- Compare page supports multi‑ticker, multi‑model classification with CSV export.
- Predictor page offers classification thresholding and diagnostics (ROC/PR AUC, confusion, misses).
- One‑click Batch Evaluation is available under Train → Evaluate section to produce a consolidated CSV in `reports/`.

## Quick User Input Prediction (Train page)
Run a saved model on the latest rows for its ticker/horizon without re‑training. This is useful to sanity‑check a trained model on new data and to generate a quick forecast table.

What you provide
- Model: choose a file from `models/` (e.g., `svm_reg_BAC_h1.joblib`). The app parses the ticker (BAC), task (reg/cls), and horizon (h1) from the name.
- Optional date range: start and end dates filter the source data first. Leave end blank to use the latest.
- Rows to predict (N): a slider selecting how many of the most recent rows to show.

What the app does
1. Loads the selected model and its metadata.
2. Locates the corresponding ticker’s data from `data/` and builds the same features used during training (scaling, technical indicators, etc.).
3. Aligns features to the model’s expected schema; for regression, the target is the next‑period close (next_close).
4. Produces predictions on the filtered dataset and takes the latest N rows for display.
5. If actuals are available (i.e., not future rows), computes per‑row absolute error for regression.

Output interpretation
- Regression table columns: `Date` (bar used to create features), `Actual_next_close`, `Predicted_next_close`, and `AbsError` (|actual − predicted|).
- Classification table columns: `Date`, `Predicted_label` (e.g., up/down/stable), and where available, `Probability`/`Confidence`.
- Important: for regression, the date is the feature time. The target is the next bar’s close. So an entry dated 2025‑08‑04 compares the prediction made from 2025‑08‑04 features to the actual close on 2025‑08‑05.

Example (from the screenshot)
- Model: `svm_reg_BAC_h1.joblib`
- One row shows `Actual_next_close = 45.56` vs `Predicted_next_close = 45.81`, yielding `AbsError ≈ 0.2501`.

Notes and edge cases
- If the selected date range extends into the future, `Actual_next_close` may be missing; the app will still show predictions but can’t compute error.
- Picking a model whose ticker isn’t present in your `data/` will yield an empty result. Ensure the CSV for that ticker exists.
- This feature does not re‑fit the model or update scalers; it runs inference with the saved pipeline. For true evaluation, prefer the Holdout Evaluation section.

## License
This project is for educational/demo purposes. Not financial advice.

---

Made with ❤️ using Streamlit and scikit‑learn/XGBoost.
