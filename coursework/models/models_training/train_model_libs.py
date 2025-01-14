# Importing necessary libraries
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to load dataset
def load_dataset(file_path):

    try:
        dataset = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

# Function to split features and target
def split_features_target(dataset, target_column):

    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

# Function to train the model
def train_model(X, y):

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    print("Model training complete.")
    return model

# Function to evaluate the model
def evaluate_model(model, X, y):

    predictions = model.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1 Score": f1_score(y, predictions)
    }
    print("Model evaluation metrics:", metrics)
    return metrics

# Function to save the model
def save_model(model, file_path):
    """
    Save the trained model to a file.

    Args:
        model: Trained model.
        file_path (str): File path to save the model.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved at: {file_path}")

# Main execution
if __name__ == "__main__":
    dataset_path = '../../datasets/normalized_shuffled_train.csv'
    model_save_path = '../model_lib_weights.pkl'
    target_column = 'TenYearCHD'

    # Step 1: Load dataset
    dataset = load_dataset(dataset_path)

    # Step 2: Split dataset into features and target
    X_train, y_train = split_features_target(dataset, target_column)

    # Step 3: Train the model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate the model
    evaluate_model(model, X_train, y_train)

    # Step 5: Save the trained model
    save_model(model, model_save_path)
