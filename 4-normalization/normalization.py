import pandas as pd
import matplotlib.pyplot as plt

file_path = "ex1data2.txt"
data = pd.read_csv(file_path, header=None, names=["Скорость Оборота Двигателя", "Количество Передач", "Цена"])


def normalize_max(data):
    """
    Формула: x'_j = x_j / max(x_j)
    """
    return data / data.max()


def normalize_range(data):
    """
    Формула: x'_j = (x_j - mean(x_j)) / (max(x_j) - min(x_j))
    """
    mean = data.mean()
    range_ = data.max() - data.min()
    return (data - mean) / range_


def normalize_std(data):
    """
    Формула: x'_j = (x_j - mean(x_j)) / std(x_j)
    """
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


# Вычисление среднего и стандартного отклонения явно по определению
def calculate_mean(data):
    """
    Среднее значение: mean = sum(x_i) / N
    """
    return sum(data) / len(data)


def calculate_std(data, mean):
    """
    Стандартное отклонение: std = sqrt(sum((x_i - mean)^2) / N)
    """
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5


# Признаки и целевая переменная
features = data.iloc[:, :-1]
price = data.iloc[:, -1]

# Добавление целевой переменной в список для обработки
columns_to_process = features.join(price).columns

# Вычисление среднего и СКО по каждой колонке
for column in columns_to_process:
    column_data = data[column]
    mean_by_def = calculate_mean(column_data)
    std_by_def = calculate_std(column_data, mean_by_def)

    # Использование встроенных функций pandas
    #mean_builtin = column_data.mean()
    #std_builtin = column_data.std()

    print(f"Колонка: {column}")
    print(f"Среднее (по определению): {mean_by_def}")
    #print(f"Среднее (встроенная функция): {mean_builtin}")
    print(f"СКО (по определению): {std_by_def}")
    #print(f"СКО (встроенная функция): {std_builtin}")
    print("-" * 40)

# Нормализация
norm_max = features.apply(normalize_max, axis=0)
norm_range = features.apply(normalize_range, axis=0)
norm_std = features.apply(normalize_std, axis=0)

# Построение графиков
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.scatter(features.iloc[:, 0], price, color="blue")
plt.xlabel("Скорость Оборота Двигателя")
plt.ylabel("Цена")
plt.title("Исходные Признаки")
plt.grid()

plt.subplot(2, 2, 2)
plt.scatter(norm_max.iloc[:, 0], price, color="green")
plt.xlabel("Скорость Оборота Двигателя (норм.)")
plt.ylabel("Цена")
plt.title("Нормировка 1 (max)")
plt.grid()

plt.subplot(2, 2, 3)
plt.scatter(norm_range.iloc[:, 0], price, color="red")
plt.xlabel("Скорость Оборота Двигателя (норм.)")
plt.ylabel("Цена")
plt.title("Нормировка 2 (max - min)")
plt.grid()

plt.subplot(2, 2, 4)
plt.scatter(norm_std.iloc[:, 0], price, color="purple")
plt.xlabel("Скорость Оборота Двигателя (норм.)")
plt.ylabel("Цена")
plt.title("Нормировка 3 (стандартное отклонение)")
plt.grid()

plt.tight_layout()
plt.savefig("normalized_features.png")
