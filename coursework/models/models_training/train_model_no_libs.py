# train_manual.py
# -----------------------------------------------
# Manual gradient descent for logistic regression,
# capturing per-epoch training history and saving plots.
# -----------------------------------------------

import math
import matplotlib.pyplot as plt
import numpy as np

# Function to load dataset (manually)
def load_dataset(file_path):
    """
    Reads the CSV file line by line and converts values to float.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    headers = lines[0].strip().split(',')
    data = [list(map(float, line.strip().split(','))) for line in lines[1:]]
    return headers, data

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def train_manual(X, y, epochs=2000, learning_rate=0.05):
    """
    Performs a manual gradient descent for logistic regression.
    Tracks the MSE and Accuracy at each epoch.
    Returns:
        weights: final learned weights (bias + coefficients)
        errors: list of MSE values per epoch
        accuracies: list of accuracy values per epoch
    """
    # Convert X, y to numpy for easier manipulation
    X = np.array(X)
    y = np.array(y)

    # Add a bias term: shape -> (n_samples, n_features+1)
    ones_col = np.ones((X.shape[0], 1))
    X = np.hstack((ones_col, X))  # [ [1, x1, x2, ...], [1, x1, x2, ...], ...]

    # Initialize weights
    weights = np.zeros(X.shape[1])

    errors = []
    accuracies = []

    for epoch in range(epochs):
        # Forward pass
        z = X.dot(weights)                # shape (n_samples,)
        predictions = 1.0 / (1.0 + np.exp(-z))  # sigmoid

        # MSE for tracking (though cross-entropy is typical)
        error = np.mean((predictions - y) ** 2)
        errors.append(error)

        # Accuracy
        predicted_labels = (predictions >= 0.5).astype(int)
        accuracy = np.mean(predicted_labels == y)
        accuracies.append(accuracy)

        # Compute gradient
        # derivative of cost wrt weights = X.T * (predictions - y) / n_samples
        # but here, the cost is MSE, so gradient differs slightly from CE-based logistic
        # We'll keep it consistent with MSE for the sake of "matching" the scikit approach above
        diff = (predictions - y)
        grad = (X.T.dot(diff)) / X.shape[0]

        # Weight update
        weights -= learning_rate * grad

        # (Optional) print progress
        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}, MSE={error:.4f}, Acc={accuracy:.4f}")

    return weights, errors, accuracies

def evaluate_manual(weights, X, y):
    """
    Evaluates final metrics given learned weights.
    Returns a dictionary of Accuracy, Precision, Recall, F1 Score
    """

    # Convert to numpy
    X = np.array(X)
    y = np.array(y)

    # Re-insert bias column
    ones_col = np.ones((X.shape[0], 1))
    X = np.hstack((ones_col, X))

    # Predictions
    z = X.dot(weights)
    probs = 1.0 / (1.0 + np.exp(-z))
    y_pred = (probs >= 0.5).astype(int)

    # Compute metrics
    # We'll replicate what scikit-learn does
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred)
    }

def plot_training_history(errors, accuracies, error_plot_path, accuracy_plot_path):
    """
    Saves two separate plots:
      1) MSE vs. Epoch
      2) Accuracy vs. Epoch
    """
    # MSE vs. Epoch
    plt.figure(figsize=(7,5))
    plt.plot(range(len(errors)), errors, label='MSE')
    plt.title('Training Error (MSE) vs. Epoch - Manual')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(error_plot_path)
    plt.close()

    # Accuracy vs. Epoch
    plt.figure(figsize=(7,5))
    plt.plot(range(len(accuracies)), accuracies, label='Accuracy', color='orange')
    plt.title('Training Accuracy vs. Epoch - Manual')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(accuracy_plot_path)
    plt.close()

if __name__ == "__main__":
    # Adjust paths as needed
    dataset_path = '../../datasets/normalized_shuffled_train.csv'
    weights_save_path = '../model_manual_weights.txt'
    error_plot_path = '../manual_error_plot.png'
    accuracy_plot_path = '../manual_accuracy_plot.png'

    EPOCHS = 2000
    LR = 0.05

    # Step 1: Load dataset
    headers, data = load_dataset(dataset_path)

    # Step 2: Identify feature indices and target index
    target_index = headers.index('TenYearCHD')
    feature_indices = [i for i in range(len(headers)) if i != target_index]

    # Prepare X and y
    X_train = [[row[i] for i in feature_indices] for row in data]
    y_train = [row[target_index] for row in data]

    # Step 3: Train (manual gradient descent)
    print("Starting manual gradient descent training...")
    weights, errors, accuracies = train_manual(X_train, y_train, epochs=EPOCHS, learning_rate=LR)

    # Step 4: Plot training history (error & accuracy)
    plot_training_history(errors, accuracies, error_plot_path, accuracy_plot_path)

    # Step 5: Evaluate final model
    metrics = evaluate_manual(weights, X_train, y_train)
    print("Final training metrics (manual):", metrics)

    # Step 6: Save the model weights (bias + coefficients)
    with open(weights_save_path, 'w') as f:
        f.write(','.join(map(str, weights)))
    print(f"Trained weights saved at: {weights_save_path}")
    print("Done.")
