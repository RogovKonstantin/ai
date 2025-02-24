import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Load the raw insurance dataset
df = pd.read_csv("insurance.csv")

# Store a copy before shuffling for visualization
df_before_shuffle = df.copy()

# Assign an index column before shuffling to track movement
df_before_shuffle["original_index"] = list(range(len(df)))

# -------------------------------
# 1. Label-encode categorical columns
# -------------------------------
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
unique_regions = df['region'].unique()
region_mapping = {region: i for i, region in enumerate(unique_regions)}
df['region'] = df['region'].map(region_mapping)

# -------------------------------
# 2. Normalize numeric features
# -------------------------------
for col in ['age', 'bmi', 'children']:
    mean_val = df[col].mean()
    std_val = df[col].std()
    df[col] = (df[col] - mean_val) / std_val

# -------------------------------
# 3. Log-transform the target
# -------------------------------
df['charges'] = df['charges'].apply(lambda x: x if x <= 0 else np.log(x))
df_before_shuffle['charges'] = df_before_shuffle['charges'].apply(lambda x: x if x <= 0 else np.log(x))

# -------------------------------
# 4. Manually Shuffle the Data (Fisher-Yates Algorithm)
# -------------------------------
random.seed(42)  # Ensure reproducibility
df_list = df.values.tolist()

for i in range(len(df_list) - 1, 0, -1):
    j = random.randint(0, i)
    df_list[i], df_list[j] = df_list[j], df_list[i]

df = pd.DataFrame(df_list, columns=df.columns)

df["shuffled_index"] = list(range(len(df)))

# Split manually into train and test sets (70% train, 30% test)
split_point = int(0.7 * len(df))
train_df = df.iloc[:split_point].copy()
test_df = df.iloc[split_point:].copy()

# Save as CSV
train_df.to_csv("train_numeric.csv", index=False)
test_df.to_csv("test_numeric.csv", index=False)

# -------------------------------
# Improved Visualization: Shuffle Effect with Line Plots
# -------------------------------
plt.figure(figsize=(12, 8))

# Plot before shuffle as a line plot
plt.subplot(2, 1, 1)
plt.plot(df_before_shuffle["original_index"], df_before_shuffle["charges"], color="blue", linestyle="-", linewidth=0.5)
plt.title("До перемешивания (Charges)")
plt.xlabel("Индекс (до shuffle)")
plt.ylabel("Charges")

# Plot after shuffle as a line plot
plt.subplot(2, 1, 2)
plt.plot(df["shuffled_index"], df["charges"], color="red", linestyle="-", linewidth=0.5)
plt.title("После перемешивания (Charges)")
plt.xlabel("Индекс (после shuffle)")
plt.ylabel("Charges")

plt.tight_layout()
plt.savefig("shuffle_effect_lineplot.png")
plt.close()

# -------------------------------
# Visualization: Feature Distribution Before and After Preprocessing
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 10))

columns_to_plot = ['age', 'bmi']
for i, col in enumerate(columns_to_plot):
    ax = axes[i]
    ax.hist(df_before_shuffle[col], bins=20, alpha=0.5, label="До нормализации")
    ax.hist(df[col], bins=20, alpha=0.5, label="После нормализации", color='red')
    ax.set_title(f"Feature: {col}")
    ax.legend()

plt.tight_layout()
plt.savefig("preprocessing_effect.png")
plt.close()

print("Preprocessing complete. Numeric files saved:")
print(" - train_numeric.csv")
print(" - test_numeric.csv")
print(" - shuffle_effect_lineplot.png (clearer shuffle visualization with line plots)")
print(" - preprocessing_effect.png (feature preprocessing visualization)")
