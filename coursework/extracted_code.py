import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Load and preprocess data
df = pd.read_csv('framingham_heart_disease.csv')
data = df.drop(['education'], axis=1)

# Fill missing values with mean
mean_fill_cols = ["cigsPerDay", "BPMeds", "totChol", "BMI", "glucose", "heartRate"]
for col in mean_fill_cols:
    data[col].fillna(round(data[col].mean()), inplace=True)

# Normalize selected columns
cols_to_be_normalized = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
cols_not_to_be_normalized = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "TenYearCHD"]

normalized = normalize(data[cols_to_be_normalized])
boolean = data[cols_not_to_be_normalized]
df_normalized = pd.DataFrame(normalized, columns=cols_to_be_normalized)
df_boolean = pd.DataFrame(boolean, columns=cols_not_to_be_normalized)
df_final = df_normalized.merge(df_boolean, left_index=True, right_index=True)

# Split dataset
X = df_final.drop("TenYearCHD", axis=1)
Y = df_final["TenYearCHD"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.2)

# Train logistic regression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict and compute metrics
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Training MSE:", train_mse)
print("Test MSE:", test_mse)
