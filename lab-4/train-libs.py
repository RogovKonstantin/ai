import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Загрузка данных из файла
file_path = 'ex2data1.txt'  # Укажите путь к вашему файлу
data = pd.read_csv(file_path, header=None, names=["Vibration", "Rotation", "Label"])

# Разделение данных на признаки (X) и целевую переменную (y)
X = data[["Vibration", "Rotation"]].values
y = data["Label"].values

# Визуализация исходных данных
plt.figure(figsize=(8, 6))
# Цикл для отображения каждой категории данных разными цветами и маркерами
for label, color, marker in zip([0, 1], ['red', 'blue'], ['o', 'x']):
    subset = data[data["Label"] == label]
    plt.scatter(subset["Vibration"], subset["Rotation"], label=f"Класс {label}", c=color, marker=marker)
plt.xlabel("Вибрация")
plt.ylabel("Ротация")
plt.legend()
plt.title("Данные двигателя")
plt.savefig("categories.png")  # Сохранение графика

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Добавление полиномиальных признаков степени 2 для захвата нелинейных зависимостей
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Обучение модели логистической регрессии с увеличенным количеством итераций для сходимости
model = LogisticRegression(max_iter=1000)
model.fit(X_train_poly, y_train)

# Предсказание меток на тестовой выборке
y_pred = model.predict(X_test_poly)
# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)

print(f"Точность: {accuracy:.2f}")
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred))

# Визуализация границы принятия решений
# Определение диапазона значений для построения сетки
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# Создание сетки точек
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
# Преобразование сетки с помощью полиномиальных признаков
Z = model.predict(poly.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
# Отображение области, соответствующей каждому классу
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
# Повторное отображение исходных данных поверх границы решений
for label, color, marker in zip([0, 1], ['red', 'blue'], ['o', 'x']):
    subset = data[data["Label"] == label]
    plt.scatter(subset["Vibration"], subset["Rotation"], label=f"Класс {label}", c=color, marker=marker)
plt.xlabel("Вибрация")
plt.ylabel("Ротация")
plt.legend()
plt.title("Граница принятия решений")
plt.savefig("libs_decision_boundary.png")  # Сохранение графика границы решений