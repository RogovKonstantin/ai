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

# Replace with your actual data file path
X, Y = read_data("framingham_heart_disease.csv")

# Instantiate the model
regressor = LogRegression(epochs=1000, lr=0.01)

# Train the model
regressor.fit(X, Y)

# Predict on the same dataset (for testing purposes)
Y_pred = regressor.predict(X)

# Calculate accuracy manually
correctly_classified = sum(1 for i in range(len(Y_pred)) if Y[i] == Y_pred[i])
accuracy = (correctly_classified / len(Y_pred)) * 100

print("Accuracy:", accuracy)
