import random
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('../dataset.csv')

numerical_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
dataset[numerical_features] = dataset[numerical_features].fillna(dataset[numerical_features].mean())

categorical_features = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
dataset[categorical_features] = dataset[categorical_features].fillna(dataset[categorical_features].mode().iloc[0])


def normalize(df, columns):
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0
    return df


def shuffle_data(dataframe, seed=42):
    random.seed(seed)
    lines = dataframe.values.tolist()
    for i in range(len(lines) - 1, 0, -1):
        j = random.randint(0, i)
        lines[i], lines[j] = lines[j], lines[i]
    return pd.DataFrame(lines, columns=dataframe.columns)


dataset = normalize(dataset, numerical_features)

dataset = shuffle_data(dataset, seed=42)

train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)

train_data.to_csv('../normalized_shuffled_train.csv', index=False)
test_data.to_csv('../normalized_shuffled_test.csv', index=False)
