import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to load dataset
def load_dataset(file_path):
    """
    Loads the CSV dataset from the given file_path.
    """
    try:
        dataset = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

# Function to split features and target
def split_features_target(dataset, target_column):
    """
    Splits the dataset into features (X) and target (y).
    """
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    return X, y

def train_model_iterative(X, y, epochs=2000, learning_rate=0.05):
    """
    Trains a logistic regression model using SGDClassifier in an iterative manner.
    We remove regularization (penalty='none') to mimic manual gradient descent.
    We track the mean-squared-error (MSE) and accuracy at each epoch.
    """

    model = SGDClassifier(
        loss='log_loss',       # logistic regression
        penalty=None,   # no regularization
        learning_rate='constant',
        eta0=learning_rate,
        max_iter=1,       # we'll manually iterate
        warm_start=True,  # so we can call partial_fit repeatedly
        random_state=0
    )

    # Convert X, y to numpy arrays
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    errors = []
    accuracies = []

    # SGDClassifier needs the list of classes to be passed in partial_fit for the first call
    classes = np.unique(y_np)

    for epoch in range(epochs):
        # partial_fit does exactly 1 epoch over the data
        model.partial_fit(X_np, y_np, classes=classes)

        # Predictions (probabilities) for MSE
        pred_probs = model.predict_proba(X_np)[:, 1]
        mse = np.mean((pred_probs - y_np) ** 2)
        errors.append(mse)

        # Accuracy
        y_pred = model.predict(X_np)
        acc = accuracy_score(y_np, y_pred)
        accuracies.append(acc)

        # You could print intermediate progress if desired
        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}, MSE={mse:.4f}, Accuracy={acc:.4f}")

    return model, errors, accuracies

def evaluate_model(model, X, y):
    """
    Evaluates final metrics on the entire dataset (training set here).
    """
    y_pred = model.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred)
    }
    return metrics

def save_model(model, file_path):
    """
    Saves the trained model to a file in pickle format.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved at: {file_path}")

def plot_training_history(errors, accuracies, error_plot_path, accuracy_plot_path):
    """
    Saves two separate plots:
      1) MSE vs. Epoch
      2) Accuracy vs. Epoch
    """
    # -- MSE vs. Epoch
    plt.figure(figsize=(7,5))
    plt.plot(range(len(errors)), errors, label='MSE')
    plt.title('Training Error (MSE) vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(error_plot_path)
    plt.close()

    # -- Accuracy vs. Epoch
    plt.figure(figsize=(7,5))
    plt.plot(range(len(accuracies)), accuracies, label='Accuracy', color='orange')
    plt.title('Training Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(accuracy_plot_path)
    plt.close()

# Main execution block (scikit-learn iterative training)
if __name__ == "__main__":
    # Adjust paths as needed
    dataset_path = '../../datasets/normalized_shuffled_train.csv'
    model_save_path = '../model_lib_weights.pkl'

    # Where to save the training history plots
    error_plot_path = '../sklearn_error_plot.png'
    accuracy_plot_path = '../sklearn_accuracy_plot.png'

    target_column = 'TenYearCHD'
    EPOCHS = 2000
    LR = 0.05

    # Step 1: Load dataset
    dataset = load_dataset(dataset_path)

    # Step 2: Split dataset into features and target
    X_train, y_train = split_features_target(dataset, target_column)

    # Step 3: Train the model iteratively
    print("Starting scikit-learn iterative training...")
    model, errors, accuracies = train_model_iterative(X_train, y_train, EPOCHS, LR)

    # Step 4: Plot training history (error & accuracy)
    plot_training_history(errors, accuracies, error_plot_path, accuracy_plot_path)

    # Step 5: Evaluate final model on training data
    metrics = evaluate_model(model, X_train, y_train)
    print("Final training metrics (scikit-learn):", metrics)

    # Step 6: Save the trained model
    save_model(model, model_save_path)
    print("Done.")