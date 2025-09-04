import streamlit as st
import pandas as pd
import joblib
import os
from src import model as model_module
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

st.title('Evaluate Model on Holdout Data')

# --- Model selection ---
model_dir = 'models'
model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
model_choice = st.selectbox('Select a trained model:', model_files)

# --- Data selection ---
data_dir = 'data'
data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
data_choice = st.selectbox('Select a dataset:', data_files)

# --- Load data ---
df = pd.read_csv(os.path.join(data_dir, data_choice))

# --- Feature selection ---
default_features = [col for col in df.columns if col not in ['label', 'next_close', 'date']]
features = st.multiselect('Select features:', df.columns.tolist(), default=default_features)

# --- Task type selection ---
task_type = st.radio('Task type:', ['Classification', 'Regression'])

# --- Holdout split ---
test_size = st.slider('Holdout set size (percent)', 10, 50, 20)
train_df = df.iloc[:-int(len(df)*test_size/100)]
holdout_df = df.iloc[-int(len(df)*test_size/100):]

# --- Run evaluation ---
if st.button('Evaluate'):
    # Load model
    model_path = os.path.join(model_dir, model_choice)
    trained_model = model_module.load_model(model_path)
    X_holdout = holdout_df[features]
    if task_type == 'Classification':
        y_holdout = holdout_df['label']
        preds = trained_model.predict(X_holdout)
        # If model output needs to be shifted (as in your code)
        if hasattr(trained_model, 'predict') and (preds.min() >= 0):
            preds = preds - 1
        report = classification_report(y_holdout, preds, output_dict=True)
        st.subheader('Classification Report (Holdout Set)')
        st.json(report)
    else:
        y_holdout = holdout_df['next_close']
        preds = trained_model.predict(X_holdout)
        mae = mean_absolute_error(y_holdout, preds)
        rmse = mean_squared_error(y_holdout, preds, squared=False)
        r2 = r2_score(y_holdout, preds)
        st.subheader('Regression Metrics (Holdout Set)')
        st.write(f'MAE: {mae:.4f}')
        st.write(f'RMSE: {rmse:.4f}')
        st.write(f'R2: {r2:.4f}')
