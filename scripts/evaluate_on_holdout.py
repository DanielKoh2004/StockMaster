"""
Evaluate model performance on unseen (holdout) data.
This script demonstrates how to split data, train a model, and evaluate on a holdout set.
Modify the script for your specific use case (classification or regression).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src import model

# Example: Load your data (update path and columns as needed)
df = pd.read_csv('data/AAPL.csv')

# Specify features and label/target columns
features = [col for col in df.columns if col not in ['label', 'next_close', 'date']]
label_col = 'label'  # For classification
# target_col = 'next_close'  # For regression

# Split into train and holdout (unseen) sets (e.g., last 20% as holdout)
train_df, holdout_df = train_test_split(df, test_size=0.2, shuffle=False)

# Train model on training set (choose the function you want)
# For classification:
model_path = model.train_xgb_classifier(train_df, features, label_col=label_col, save_name='xgb_class_AAPL_holdout.joblib')
# For regression, use:
# model_path = model.train_xgb_regressor(train_df, features, target_col='next_close', save_name='xgb_reg_AAPL_holdout.joblib')

# Load the best model
trained_model = model.load_model(model_path)

# Evaluate on holdout set
X_holdout = holdout_df[features]
y_holdout = holdout_df[label_col]  # or 'next_close' for regression

# For classification:
preds = trained_model.predict(X_holdout)
preds = preds - 1  # If your model expects this transformation
from sklearn.metrics import classification_report
print('Holdout Classification Report:')
print(classification_report(y_holdout, preds))

# For regression, use:
# preds = trained_model.predict(X_holdout)
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# print('Holdout MAE:', mean_absolute_error(y_holdout, preds))
# print('Holdout RMSE:', mean_squared_error(y_holdout, preds, squared=False))
# print('Holdout R2:', r2_score(y_holdout, preds))
