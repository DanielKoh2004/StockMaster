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

## License
This project is for educational/demo purposes. Not financial advice.

---

Made with ❤️ using Streamlit and scikit‑learn/XGBoost.
