import pandas as pd

input_csv = '../dataset.csv'
output_csv = '../dataset_cleaned.csv'

data = pd.read_csv(input_csv)

mean_fill_cols = ["cigsPerDay", "BPMeds", "totChol", "BMI", "glucose", "heartRate"]

for col in mean_fill_cols:
    data[col].fillna(round(data[col].mean()), inplace=True)

cols_to_be_normalized = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
                         "education"]
for col in cols_to_be_normalized:
    data[col].fillna(round(data[col].mean()), inplace=True)

data.to_csv(output_csv, index=False)
