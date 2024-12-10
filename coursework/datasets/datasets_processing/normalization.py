import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Input and output file paths
input_csv = '../dataset.csv'
output_csv = '../dataset_normalized.csv'

# Load the dataset
data = pd.read_csv(input_csv)

# Columns to fill missing values with the mean
mean_fill_cols = ["cigsPerDay", "BPMeds", "totChol", "BMI", "glucose", "heartRate"]
for col in mean_fill_cols:
    data[col].fillna(data[col].mean(), inplace=True)

# Columns to normalize (includes some from mean_fill_cols)
cols_to_be_normalized = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
                         "education"]

# Ensure no missing values in the columns to be normalized
for col in cols_to_be_normalized:
    if data[col].isnull().any():
        data[col].fillna(data[col].mean(), inplace=True)

# Normalize specified columns using Min-Max scaling
scaler = MinMaxScaler()
data[cols_to_be_normalized] = scaler.fit_transform(data[cols_to_be_normalized])

# Save the cleaned and normalized dataset
data.to_csv(output_csv, index=False)
