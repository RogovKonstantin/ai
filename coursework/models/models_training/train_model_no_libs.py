def load_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    headers = lines[0].strip().split(',')
    data = [list(map(float, line.strip().split(','))) for line in lines[1:]]
    return headers, data


def sigmoid(z):
    return 1 / (1 + (2.71828 ** -z))


def train_manual(X, y, epochs=1000, learning_rate=0.1):
    X = [[1.0] + row for row in X]
    weights = [0.0] * len(X[0])

    for _ in range(epochs):
        for i in range(len(X)):
            z = sum(w * x for w, x in zip(weights, X[i]))
            prediction = sigmoid(z)
            error = prediction - y[i]
            for j in range(len(weights)):
                weights[j] -= learning_rate * error * X[i][j]

    return weights


headers, data = load_dataset('../../datasets/normalized_shuffled_train.csv')

feature_indices = [i for i in range(len(headers)) if headers[i] != 'TenYearCHD']
target_index = headers.index('TenYearCHD')
X_train = [[row[i] for i in feature_indices] for row in data]
y_train = [row[target_index] for row in data]

weights = train_manual(X_train, y_train, epochs=1000, learning_rate=0.1)

with open('../model_manual_weights.txt', 'w') as f:
    f.write(','.join(map(str, weights)))
