import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('datasets/dataset_final.csv')

X = data.drop(columns=['TenYearCHD'])
y = data['TenYearCHD']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

sklearn_model = SklearnLogisticRegression(random_state=0, max_iter=1000)  # Increased max_iter
sklearn_model.fit(X_train, y_train)

y_train_pred_sklearn = sklearn_model.predict(X_train)
y_test_pred_sklearn = sklearn_model.predict(X_test)

def sigmoid(z):
    z = max(min(z, 100), -100)
    return 1 / (1 + (2.718281828459045 ** -z))

def predict(X, weights):
    y_pred = []
    for x in X:
        z = sum(w * x_i for w, x_i in zip(weights, x))
        y_pred.append(1 if sigmoid(z) >= 0.5 else 0)
    return y_pred

def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    weights = [0.0] * len(X[0])

    for epoch in range(epochs):
        for i in range(len(X)):
            z = sum(w * x_i for w, x_i in zip(weights, X[i]))
            prediction = sigmoid(z)
            for j in range(len(weights)):
                weights[j] += learning_rate * (y[i] - prediction) * X[i][j]

    return weights

X_train_custom = X_train.tolist()
X_test_custom = X_test.tolist()

weights = logistic_regression(X_train_custom, y_train.tolist())

y_train_pred_custom = predict(X_train_custom, weights)
y_test_pred_custom = predict(X_test_custom, weights)

print("Sklearn Logistic Regression:")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_sklearn))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred_sklearn))
print("Training MSE:", mean_squared_error(y_train, y_train_pred_sklearn))
print("Test MSE:", mean_squared_error(y_test, y_test_pred_sklearn))

print("\nCustom Logistic Regression:")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_custom))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred_custom))
print("Training MSE:", mean_squared_error(y_train, y_train_pred_custom))
print("Test MSE:", mean_squared_error(y_test, y_test_pred_custom))

print("\nExample Predictions:")

example_rows = X_test[:10]

sklearn_example_predictions = sklearn_model.predict(example_rows)
print("\nSklearn Model Predictions:")
for i, prediction in enumerate(sklearn_example_predictions):
    print(f"Example {i+5}: Predicted = {prediction}, Actual = {y_test.iloc[i]}")

custom_example_predictions = predict(example_rows.tolist(), weights)
print("\nCustom Model Predictions:")
for i, prediction in enumerate(custom_example_predictions):
    print(f"Example {i+5}: Predicted = {prediction}, Actual = {y_test.iloc[i]}")
