import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_and_save():
    """
    1) Loads ../dataset.csv using pandas.
    2) Drops NA rows.
    3) Applies Z-score standardization to numerical features.
    4) Shuffles dataset.
    5) Splits into train/test sets and saves them to:
       - ../normalized_shuffled_train.csv
       - ../normalized_shuffled_test.csv
    6) Saves distribution comparison plot to ../standardization_comparison_all_features.png
    """

    # ------------------ Load the dataset ------------------ #
    original_dataset = pd.read_csv('../dataset.csv')

    # Drop rows with any missing values
    original_dataset = original_dataset.dropna()

    # Identify numerical and categorical features
    numerical_features = [
        'age', 'cigsPerDay', 'totChol', 'sysBP',
        'diaBP', 'BMI', 'heartRate', 'glucose'
    ]
    categorical_features = [
        'male', 'education', 'currentSmoker', 'BPMeds',
        'prevalentStroke', 'prevalentHyp', 'diabetes'
    ]

    # We will work on a copy for standardization
    dataset = original_dataset.copy()

    # --------------- Apply Z-score standardization --------------- #
    scaler = StandardScaler()
    dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])
    print("Saving scaler parameters to:", 'scaler_params.pkl')
    pickle.dump({"means": scaler.mean_, "scales": scaler.scale_}, open('scaler_params.pkl', "wb"))

    # ------------------ Shuffle the dataset ------------------ #
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # --------------- Split the dataset into train/test --------------- #
    train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)

    # --------------- Save the processed data --------------- #
    train_data.to_csv('../normalized_shuffled_train.csv', index=False)
    test_data.to_csv('../normalized_shuffled_test.csv', index=False)

    # --------------- Visualization: Compare distributions --------------- #
    rows = (len(numerical_features) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for idx, feature in enumerate(numerical_features):
        # Before standardization (original_dataset)
        sns.histplot(
            original_dataset[feature],
            bins=30,
            kde=True,
            color='blue',
            label='Before Standardization',
            stat="density",
            alpha=0.5,
            ax=axes[idx]
        )
        # After standardization (dataset)
        sns.histplot(
            dataset[feature],
            bins=30,
            kde=True,
            color='orange',
            label='After Standardization',
            stat="density",
            alpha=0.5,
            ax=axes[idx]
        )
        axes[idx].set_title(f'Distribution of "{feature}"')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Density')
        axes[idx].legend()

    # Remove any extra subplots
    for ax in axes[len(numerical_features):]:
        fig.delaxes(ax)

    fig.suptitle('Comparison of Distributions Before and After Standardization',
                 fontsize=16, y=0.92)
    plt.tight_layout()
    plt.savefig('../standardization_comparison_all_features.png')
    plt.close()

# Add the main block to execute the function
if __name__ == "__main__":
    preprocess_and_save()
