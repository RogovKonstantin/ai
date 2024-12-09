class LogRegression:
    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.lr = lr
        self.W = None
        self.b = None

    def fit(self, X, Y):
        self.m = len(Y)  # Number of samples
        self.n = len(X[0])  # Number of features
        self.W = [0.0] * self.n  # Initialize weights to 0
        self.b = 0.0  # Initial bias is 0

        # Training loop (gradient descent)
        for i in range(self.epochs):
            self.update_weights(X, Y)

        return self

    def update_weights(self, X, Y):
        A = [self.sigmoid(self.dot_product(X[i], self.W) + self.b) for i in range(self.m)]

        # Calculate gradients
        tmp = [A[i] - Y[i] for i in range(self.m)]  # Error
        D_W = [0.0] * self.n
        D_b = 0.0

        for i in range(self.m):
            for j in range(self.n):
                D_W[j] += (tmp[i] / self.m) * X[i][j]
            D_b += tmp[i] / self.m

        # Update weights and bias
        for j in range(self.n):
            self.W[j] -= self.lr * D_W[j]
        self.b -= self.lr * D_b

    def predict(self, X):
        Z = [self.sigmoid(self.dot_product(X[i], self.W) + self.b) for i in range(len(X))]
        return [1 if z > 0.5 else 0 for z in Z]

    def sigmoid(self, x):
        if x >= 0:
            return 1 / (1 + (2.71828 ** -x))
        else:
            exp_x = 2.71828 ** x
            return exp_x / (exp_x + 1)

    def dot_product(self, X_row, W):
        return sum(X_row[j] * W[j] for j in range(len(W)))

    def mean_squared_error(self, predictions, Y):
        mse = sum((predictions[i] - Y[i]) ** 2 for i in range(len(Y))) / len(Y)
        return mse

    def accuracy(self, predictions, Y):
        correct = sum(1 for i in range(len(predictions)) if predictions[i] == Y[i])
        return correct / len(Y)

# Data reading and preprocessing
def read_data(file_path):
    X = []
    Y = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header = lines[0].strip().split(',')  # Read header line
    for line in lines[1:]:  # Skip header
        values = line.strip().split(',')

        # Attempt to convert each value to float; skip rows with 'NA' or invalid values
        try:
            features = [float(val) if val != 'NA' else 0.0 for val in values[:-1]]  # Default to 0 for 'NA'
            label = int(values[-1])  # Last column as label
            X.append(features)
            Y.append(label)
        except ValueError:
            continue  # Skip the row if there was any issue converting

    return X, Y

# Split data into training and testing sets
def train_test_split(X, Y, test_size=0.2):
    indices = list(range(len(X)))
    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]

    return X_train, X_test, Y_train, Y_test

# Main script
X, Y = read_data("../dataset/dataset.csv")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Instantiate and train the model
regressor = LogRegression(epochs=1000, lr=0.01)
regressor.fit(X_train, Y_train)

# Predict and compute metrics
Y_train_pred = regressor.predict(X_train)
Y_test_pred = regressor.predict(X_test)

train_accuracy = regressor.accuracy(Y_train_pred, Y_train)
test_accuracy = regressor.accuracy(Y_test_pred, Y_test)
train_mse = regressor.mean_squared_error(Y_train_pred, Y_train)
test_mse = regressor.mean_squared_error(Y_test_pred, Y_test)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Training MSE:", train_mse)
print("Test MSE:", test_mse)
