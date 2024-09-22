import pandas as pd
from sklearn.model_selection import train_test_split

csv_file_path = 'datasets/dataset_modified.csv'
data = pd.read_csv(csv_file_path, encoding='ISO-8859-1', usecols=['Calories Burn', 'Actual Weight', 'Age', 'Gender', 'Duration', 'Heart Rate', 'BMI'])

X = data.drop('Calories Burn', axis=1)
Y = data['Calories Burn']

X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=42)

DataFrame_train = pd.DataFrame(X_train)
DataFrame_train['Calories Burn'] = Y_train.values

DataFrame_valid = pd.DataFrame(X_val)
DataFrame_valid['Calories Burn'] = Y_val.values

DataFrame_test = pd.DataFrame(X_test)
DataFrame_test['Calories Burn'] = Y_test.values

DataFrame_train.to_csv('parts/train.csv', index=False)
DataFrame_valid.to_csv('parts/validation.csv', index=False)
DataFrame_test.to_csv('parts/test.csv', index=False)