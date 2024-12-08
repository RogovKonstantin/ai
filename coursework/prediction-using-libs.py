import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

df = pd.read_csv('C:/projects/python/pythonProject/dataset/exercise_dataset.csv')

df.drop('ID', axis=1, inplace=True)
df['Exercise'] = df['Exercise'].apply(lambda x: int(x.split()[1]))

df1 = pd.get_dummies(df, columns=['Gender', 'Weather Conditions', 'Exercise'], drop_first=True)

x1 = df1.drop('Actual Weight', axis=1)
y1 = df1['Actual Weight']

x1 = x1.select_dtypes(include=[np.number])

scaler = StandardScaler()
x1 = scaler.fit_transform(x1)

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=11)

model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"Mean Squared Error (Train) with Keras: {mse_train}")
print(f"Mean Squared Error (Test) with Keras: {mse_test}")
