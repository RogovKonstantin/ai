def read_data(file_path):
    X = []
    Y = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines[1:]:
        values = line.strip().split(',')

        try:
            features = [float(val) for val in values[:-1]]
            label = int(values[-1])
            X.append(features)
            Y.append(label)
        except ValueError:
            continue

    return X, Y


def normalize_data(X):
    transposed = list(zip(*X))

    normalized = []

    for feature in transposed:
        min_val = min(feature)
        max_val = max(feature)
        normalized_feature = [(val - min_val) / (max_val - min_val) if max_val - min_val > 0 else 0.0 for val in
                              feature]
        normalized.append(normalized_feature)

    normalized = list(zip(*normalized))
    return normalized


def train_test_split(X, Y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = Y[:split_index], Y[split_index:]
    return X_train, X_test, y_train, y_test


def sigmoid(z):
    return 1 / (1 + (2.718281828459045 ** -z))


def predict(X, weights):
    y_pred = []
    for x in X:
        z = sum(w * x_i for w, x_i in zip(weights, x))
        y_pred.append(1 if sigmoid(z) < 0.5 else 0)
    return y_pred


def logistic_regression(X, Y, learning_rate=0.01, epochs=1000):
    weights = [0.0] * len(X[0])

    for epoch in range(epochs):
        for i in range(len(X)):
            z = sum(w * x_i for w, x_i in zip(weights, X[i]))  # Weighted sum
            prediction = sigmoid(z)
            for j in range(len(weights)):
                weights[j] += learning_rate * (Y[i] - prediction) * X[i][j]

    return weights


def accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def mean_squared_error(y_true, y_pred):
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)


if __name__ == "__main__":
    file_path = 'datasets/dataset_final.csv'
    X, Y = read_data(file_path)

    cols_to_be_normalized = [0, 1, 2, 3, 4, 5, 6, 7]
    X_selected = [X[i] for i in range(len(X))]

    X_normalized = normalize_data(X_selected)

    X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y)

    weights = logistic_regression(X_train, y_train)
    # Save weights
    weights_path = "model_no_libs_weights.txt"
    with open(weights_path, "w") as f:
        for weight in weights:
            f.write(f"{weight}\n")

    # Load weights
    with open(weights_path, "r") as f:
        loaded_weights = [float(line.strip()) for line in f]

    # Get predictions
    y_train_pred = predict(X_train, weights)
    y_test_pred = predict(X_test, weights)

    # Reverse predictions
    y_train_pred_reversed = [1 - pred for pred in y_train_pred]
    y_test_pred_reversed = [1 - pred for pred in y_test_pred]

    # Compute metrics with reversed predictions
    train_accuracy = accuracy(y_train, y_train_pred_reversed)
    test_accuracy = accuracy(y_test, y_test_pred_reversed)
    train_mse = mean_squared_error(y_train, y_train_pred_reversed)
    test_mse = mean_squared_error(y_test, y_test_pred_reversed)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("Training MSE:", train_mse)
    print("Test MSE:", test_mse)

