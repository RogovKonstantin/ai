import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Input and output paths
input_csv = '../dataset.csv'
output_csv = '../dataset_normalized.csv'

# Load dataset
data = pd.read_csv(input_csv)

# Fill missing values
binary_cols = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
for col in binary_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

continuous_cols = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
for col in continuous_cols:
    data[col].fillna(data[col].mean(), inplace=True)

ordinal_cols = ["education"]
for col in ordinal_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Normalize continuous variables
scaler = MinMaxScaler()
data[continuous_cols] = scaler.fit_transform(data[continuous_cols])

# Save processed dataset
data.to_csv(output_csv, index=False)
