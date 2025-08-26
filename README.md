# StockMaster

StockMaster is a modern, interactive Streamlit app for stock, crypto, and commodity price prediction and analysis. It uses machine learning models (XGBoost) to forecast price direction and magnitude, and provides a visually rich dashboard for exploring predictions, historical data, and model performance.

## Features
- **Dynamic Data:** Fetches the latest prices online (via yfinance) with fallback to local CSVs per asset.
- **Predictions:** Shows both regression (price) and classification (direction) predictions for each asset.
- **Visual Dashboard:** Home, Assets, Predictor, Train, Models, and About pages with sticky navbar and branding.
- **Stock/Crypto/Commodity Support:** Works for stocks, cryptocurrencies, and futures/metals.
- **Model Training:** Train new XGBoost models directly from the UI.
- **Logos & Tooltips:** Asset logos and helpful tooltips for predictions.
- **Customizable Theme:** Easily adjust theme and branding.

## Installation
1. Clone the repository or download the source code.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Place your own single-ticker CSVs in the `data/` folder for offline fallback.

## Usage
Run the app from the project root:
```bash
python -m streamlit run src/app.py
```

## Folder Structure
```
StockMaster/
├── data/           # Single-ticker CSVs and historical data
├── models/         # Trained XGBoost model files (.joblib)
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
- See `requirements.txt` for all dependencies (Streamlit, yfinance, xgboost, pandas, plotly, etc.)

## Notes
- The app will attempt to fetch the latest data online for each ticker. If unavailable, it will use the corresponding CSV in `data/`.
- Models must be trained and saved in the `models/` folder to enable predictions.
- For best results, ensure your local CSVs and models are up to date.

## License
This project is for educational/demo purposes. Not financial advice.

---

Made with ❤️ using Streamlit and XGBoost.
