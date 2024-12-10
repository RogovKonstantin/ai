import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
df_final = pd.concat([df_normalized, df_boolean], axis=1)

# Split into features and target
X = df_final.drop("TenYearCHD", axis=1)
Y = df_final["TenYearCHD"]

# Use the entire dataset for evaluation
test_cases = X.values
actual_values = Y.values

# === Predictions with Libraries Model ===
libs_model = load(libs_model_path)
test_cases_df = pd.DataFrame(test_cases, columns=X.columns)  # Ensure proper feature names
libs_predictions = libs_model.predict(test_cases_df)

# === Predictions with No Libraries Model ===
with open(no_libs_weights_path, "r") as f:
    no_libs_weights = [float(line.strip()) for line in f]
test_cases_no_libs = normalize_data(test_cases.tolist())
no_libs_predictions = predict(test_cases_no_libs, no_libs_weights)

# Evaluate both models
libs_accuracy = accuracy_score(actual_values, libs_predictions)
no_libs_accuracy = accuracy_score(actual_values, no_libs_predictions)

libs_conf_matrix = confusion_matrix(actual_values, libs_predictions)
no_libs_conf_matrix = confusion_matrix(actual_values, no_libs_predictions)

libs_report = classification_report(actual_values, libs_predictions, output_dict=True)
no_libs_report = classification_report(actual_values, no_libs_predictions, output_dict=True)

# Save results to Excel
report_path = "datasets/model_comparison_report.xlsx"
comparison_results = pd.DataFrame({
    "Actual": actual_values,
    "With Libs Predictions": libs_predictions,
    "No Libs Predictions": no_libs_predictions
})

with pd.ExcelWriter(report_path) as writer:
    # Save predictions
    comparison_results.to_excel(writer, sheet_name="Predictions", index=False)
    # Save classification reports
    pd.DataFrame(libs_report).transpose().to_excel(writer, sheet_name="Libs Model Report")
    pd.DataFrame(no_libs_report).transpose().to_excel(writer, sheet_name="No Libs Model Report")
    # Save confusion matrices
    pd.DataFrame(libs_conf_matrix, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_excel(writer, sheet_name="Libs Confusion Matrix")
    pd.DataFrame(no_libs_conf_matrix, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_excel(writer, sheet_name="No Libs Confusion Matrix")

print(f"Excel report generated at: {report_path}")
