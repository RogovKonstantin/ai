import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

train_dataset = pd.read_csv('../../datasets/normalized_shuffled_train.csv')

X_train = train_dataset.drop(columns=['TenYearCHD'])
y_train = train_dataset['TenYearCHD']

model_lib = LogisticRegression()
model_lib.fit(X_train, y_train)

with open('../model_lib_weights.pkl', 'wb') as f:
    pickle.dump(model_lib, f)
