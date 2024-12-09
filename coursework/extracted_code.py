import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

data = pd.read_csv('../dataset/dataset_final.csv')

cols_to_be_normalized = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "education"]
cols_not_to_be_normalized = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
                             "TenYearCHD"]

normalized = normalize(data[cols_to_be_normalized])
boolean = data[cols_not_to_be_normalized]
df_normalized = pd.DataFrame(normalized, columns=cols_to_be_normalized)
df_boolean = pd.DataFrame(boolean, columns=cols_not_to_be_normalized)
df_final = df_normalized.merge(df_boolean, left_index=True, right_index=True)

X = df_final.drop("TenYearCHD", axis=1)
Y = df_final["TenYearCHD"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.2)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

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
