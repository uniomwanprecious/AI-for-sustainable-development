import pandas as pd
import numpy as np
import joblib # Used for saving the final model

# --- 0. Essential Imports for ML ---
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# --- Step 1, 2, 3: Load, Merge, Create Target, and Clean Data ---
print("Running Steps 1-3: Load, Merge, Create Target, and Clean Data...")
try:
    # Load the files
    features_df = pd.read_csv('dengue_features_train.csv')
    labels_df = pd.read_csv('dengue_labels_train.csv')

    # Merge the DataFrames
    combined_df = pd.merge(features_df, labels_df, on=['city', 'year', 'weekofyear'], how='inner')

    # Create Binary Target (Step 2)
    # Define a High_Outbreak as a week above the 75th percentile of total cases
    outbreak_threshold = combined_df['total_cases'].quantile(0.75)
    combined_df['High_Outbreak'] = (combined_df['total_cases'] > outbreak_threshold).astype(int)

    # Handle Missing Values (Step 3)
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.fillna(method='bfill', inplace=True)
    print(f"Data Cleaning Complete. Total missing values: {combined_df.isnull().sum().sum()}")

except FileNotFoundError:
    print("ERROR: One or both CSV files not found. Check file names and folder location.")
    exit()


# --- Step 4 & 5: Separate Data by City and Define X (Features) / Y (Target) ---
print("\nRunning Step 4 & 5: Separate Data and Define X/Y...")
sj_df = combined_df[combined_df['city'] == 'sj'].copy()
iq_df = combined_df[combined_df['city'] == 'iq'].copy()

# Drop non-feature and non-target columns for both X sets
drop_cols = ['city', 'week_start_date', 'total_cases', 'High_Outbreak']
sj_X = sj_df.drop(columns=drop_cols)
sj_y = sj_df['High_Outbreak']
iq_X = iq_df.drop(columns=drop_cols)
iq_y = iq_df['High_Outbreak']


# --- Step 6: Perform Train-Test Split (80/20) ---
print("Running Step 6: Train-Test Split...")
# San Juan Split
sj_X_train, sj_X_test, sj_y_train, sj_y_test = train_test_split(
    sj_X, sj_y, test_size=0.2, random_state=42, stratify=sj_y
)
# Iquitos Split
iq_X_train, iq_X_test, iq_y_train, iq_y_test = train_test_split(
    iq_X, iq_y, test_size=0.2, random_state=42, stratify=iq_y
)


# --- Step 7: Train Multiple Models (Decision Tree, Random Forest, Logistic Regression) ---
print("\nRunning Step 7: Model Training...")
# Initialize all three model types
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
# Logistic Regression requires the 'liblinear' solver for this type of data
lr_model = LogisticRegression(solver='liblinear', random_state=42)

# Train San Juan Models (These define the models used in evaluation)
model_sj = dt_model.fit(sj_X_train, sj_y_train)
model_sj_rf = rf_model.fit(sj_X_train, sj_y_train)
model_sj_lr = lr_model.fit(sj_X_train, sj_y_train)

# Train Iquitos Models
model_iq = dt_model.fit(iq_X_train, iq_y_train)
model_iq_rf = rf_model.fit(iq_X_train, iq_y_train)
model_iq_lr = lr_model.fit(iq_X_train, iq_y_train)
print("All six models have been successfully trained.")


# --- Step 8: Evaluate and Compare Models ---
print("\nRunning Step 8: Model Comparison...")

# San Juan Comparison
sj_models = {
    "Decision Tree": model_sj,
    "Random Forest": model_sj_rf,
    "Logistic Regression": model_sj_lr
}

print("--- San Juan Model Results ---")
sj_comparison_results = []
for name, model in sj_models.items():
    y_pred = model.predict(sj_X_test)
    f1 = f1_score(sj_y_test, y_pred, zero_division=0)
    acc = accuracy_score(sj_y_test, y_pred)
    sj_comparison_results.append((name, f1, acc))
    print(f"  {name:<20} | F1-Score: {f1:.4f} | Accuracy: {acc:.4f}")
    
# Iquitos Comparison
iq_models = {
    "Decision Tree": model_iq,
    "Random Forest": model_iq_rf,
    "Logistic Regression": model_iq_lr
}
print("\n--- Iquitos Model Results ---")
iq_comparison_results = []
for name, model in iq_models.items():
    y_pred = model.predict(iq_X_test)
    f1 = f1_score(iq_y_test, y_pred, zero_division=0)
    acc = accuracy_score(iq_y_test, y_pred)
    iq_comparison_results.append((name, f1, acc))
    print(f"  {name:<20} | F1-Score: {f1:.4f} | Accuracy: {acc:.4f}")


features_df = pd.read_csv('data/dengue_features_train.csv')
labels_df = pd.read_csv('data/dengue_labels_train.csv')
# --- Step 9: Save Best Model for Streamlit Deployment ---
print("\nRunning Step 9: Saving Deployment Model...")

# 1. Identify best model (Based on your output, Logistic Regression is best for San Juan, F1=0.2381)
# Note: We hardcode the best model name here since the evaluation determined it.
best_model_for_deployment = model_sj_lr 

# 2. Save the model and the list of feature columns (crucial for the Streamlit app)
joblib.dump(best_model_for_deployment, 'sj_logistic_model.joblib')
joblib.dump(sj_X_train.columns.tolist(), 'sj_feature_names.joblib')

print("✅ Deployment Model (sj_logistic_model.joblib) saved successfully.")