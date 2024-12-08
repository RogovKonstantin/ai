import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('framingham_heart_disease.csv')
df.head(50)
df[:].describe()
features_nan = [feature for feature in df.columns if df[feature].isnull().sum() > 1]
for feature in features_nan:
    print("{}: {}% missing values".format(feature, np.round(df[feature].isnull().mean(), 4)))
series = pd.isnull(df['cigsPerDay'])
df[series]
data = df.drop(['education'], axis=1)
data.head()
mean_cigsPerDay = round(data["cigsPerDay"].mean())
mean_BPmeds = round(data["BPMeds"].mean())
mean_totChol = round(data["totChol"].mean())
mean_BMI = round(data["BMI"].mean())
mean_glucose = round(data["glucose"].mean())
mean_heartRate = round(data["heartRate"].mean())
data['cigsPerDay'].fillna(mean_cigsPerDay, inplace=True)
data['BPMeds'].fillna(mean_BPmeds, inplace=True)
data['totChol'].fillna(mean_totChol, inplace=True)
data['BMI'].fillna(mean_BMI, inplace=True)
data['glucose'].fillna(mean_glucose, inplace=True)
data['heartRate'].fillna(mean_heartRate, inplace=True)

features_nan = [feature for feature in data.columns if data[feature].isnull().sum() > 1]

data[:].isnull().sum()
data.groupby('TenYearCHD').mean()
sns.pairplot(data[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])
cols_to_be_normalized = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
cols_not_to_be_normalized = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
                             "TenYearCHD"]

normalized = normalize(data[cols_to_be_normalized])
boolean = data[cols_not_to_be_normalized]
df_normalized = pd.DataFrame(normalized, columns=cols_to_be_normalized)
df_boolean = pd.DataFrame(boolean, columns=cols_not_to_be_normalized)
# df_final = pd.concat([df_normalized,df_boolean],axis = 1)
# df_final = df_normalized.join(df_boolean)
df_final = df_normalized.merge(df_boolean, left_index=True, right_index=True)
X = df_final.drop("TenYearCHD", axis=1)
Y = df_final["TenYearCHD"]
X = np.array(X)
Y = np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.2)
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred))
