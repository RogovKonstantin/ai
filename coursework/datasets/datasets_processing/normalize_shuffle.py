import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = pd.read_csv('../dataset.csv')

# Drop rows with any missing values
dataset = dataset.dropna()

# Identify numerical and categorical features
numerical_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
categorical_features = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

# Apply Z-score standardization to numerical features
scaler = StandardScaler()
dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

# Shuffle the dataset
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into training and testing subsets
train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)

# Save the processed data
train_data.to_csv('../normalized_shuffled_train.csv', index=False)
test_data.to_csv('../normalized_shuffled_test.csv', index=False)

# Visualization: Compare distributions before and after standardization
rows = (len(numerical_features) + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
axes = axes.flatten()

# Reload original dataset for comparison
original_dataset = pd.read_csv('../dataset.csv')

# Remove missing rows for proper comparison
original_dataset = original_dataset.dropna()

# Visualization for each numerical feature
for idx, feature in enumerate(numerical_features):
    sns.histplot(original_dataset[feature], bins=30, kde=True, color='blue', label='Before Standardization',
                 stat="density", alpha=0.5, ax=axes[idx])
    sns.histplot(dataset[feature], bins=30, kde=True, color='orange', label='After Standardization',
                 stat="density", alpha=0.5, ax=axes[idx])
    axes[idx].set_title(f'Distribution of "{feature}"')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Density')
    axes[idx].legend()

# Remove empty plots
for ax in axes[len(numerical_features):]:
    fig.delaxes(ax)

# Set overall title and save the plot
fig.suptitle('Comparison of Distributions Before and After Standardization', fontsize=16, y=0.92)
plt.tight_layout()
plt.savefig('../standardization_comparison_all_features.png')
plt.close()
