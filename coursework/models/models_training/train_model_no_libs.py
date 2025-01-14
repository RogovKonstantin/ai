# Importing required libraries
import math

# Function to load dataset
def load_dataset(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()
    headers = lines[0].strip().split(',')
    data = [list(map(float, line.strip().split(','))) for line in lines[1:]]
    return headers, data

# Sigmoid function
def sigmoid(z):

    return 1 / (1 + math.exp(-z))

# Logistic regression training function
def train_manual(X, y, epochs=1000, learning_rate=0.1):

    X = [[1.0] + row for row in X]  # Add bias term
    weights = [0.0] * len(X[0])  # Initialize weights

    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            z = sum(w * x for w, x in zip(weights, X[i]))
            prediction = sigmoid(z)
            error = prediction - y[i]
            total_error += error ** 2  # Track error for diagnostics
            for j in range(len(weights)):
                weights[j] -= learning_rate * error * X[i][j]
        if epoch % 100 == 0:  # Log progress every 100 epochs
            print(f"Epoch {epoch}/{epochs}, Error: {total_error:.4f}")
    return weights

# Main execution
if __name__ == "__main__":
    dataset_path = '../../datasets/normalized_shuffled_train.csv'
    weights_save_path = '../model_manual_weights.txt'

    # Step 1: Load dataset
    headers, data = load_dataset(dataset_path)

    # Step 2: Split features and target
    feature_indices = [i for i in range(len(headers)) if headers[i] != 'TenYearCHD']
    target_index = headers.index('TenYearCHD')
    X_train = [[row[i] for i in feature_indices] for row in data]
    y_train = [row[target_index] for row in data]

    # Step 3: Train the model
    weights = train_manual(X_train, y_train, epochs=2000, learning_rate=0.05)

    # Step 4: Save the model
    with open(weights_save_path, 'w') as f:
        f.write(','.join(map(str, weights)))
    print(f"Trained weights saved at: {weights_save_path}")
