import pandas as pd
from sklearn.preprocessing import normalize
from joblib import load
from build_model_no_libs import normalize_data, predict

# File paths for the models and dataset
dataset_path = "datasets/dataset_final.csv"
libs_model_path = "model_with_libs.joblib"
no_libs_weights_path = "model_no_libs_weights.txt"

# Load the dataset
data = pd.read_csv(dataset_path)

# Columns to normalize
cols_to_be_normalized = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "education"]
cols_not_to_be_normalized = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "TenYearCHD"]

# Normalize the data
normalized = normalize(data[cols_to_be_normalized])
boolean = data[cols_not_to_be_normalized]
df_normalized = pd.DataFrame(normalized, columns=cols_to_be_normalized)
df_boolean = pd.DataFrame(boolean, columns=cols_not_to_be_normalized)
df_final = df_normalized.merge(df_boolean, left_index=True, right_index=True)

# Split into features and target
X = df_final.drop("TenYearCHD", axis=1)
Y = df_final["TenYearCHD"]

# Extract 10 test cases from the middle of the dataset
middle_index = len(X) // 2
start_index = middle_index -800  # Adjusted indices
end_index = middle_index -700

test_cases = X.iloc[start_index:end_index].values
actual_values = Y.iloc[start_index:end_index].values

# === Predictions with Libraries Model ===
# Load the model
libs_model = load(libs_model_path)

# Pass test cases with feature names
test_cases_with_names = pd.DataFrame(test_cases, columns=X.columns)
libs_predictions = libs_model.predict(test_cases_with_names)

# === Predictions with No Libraries Model ===
# Load weights
with open(no_libs_weights_path, "r") as f:
    no_libs_weights = [float(line.strip()) for line in f]

# Normalize the test cases for no-libs model
test_cases_no_libs = normalize_data(test_cases.tolist())
no_libs_predictions = predict(test_cases_no_libs, no_libs_weights)

# === Comparison ===
print("\nComparison of Models' Predictions vs Actual Values:")
print(f"{'Case':<10}{'With Libs':<15}{'No Libs':<15}{'Actual':<10}")
for i, (libs_pred, no_libs_pred, actual) in enumerate(zip(libs_predictions, no_libs_predictions, actual_values)):
    print(f"{i + 1:<10}{libs_pred:<15}{no_libs_pred:<15}{actual:<10}")
