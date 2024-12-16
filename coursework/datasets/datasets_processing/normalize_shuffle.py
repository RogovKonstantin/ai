import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Загружаем данные из файла CSV
dataset = pd.read_csv('../dataset.csv')

# Выделяем числовые признаки для нормализации
numerical_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

# Заполняем пропущенные значения в числовых признаках средним значением по соответствующим столбцам
dataset[numerical_features] = dataset[numerical_features].fillna(dataset[numerical_features].mean())

# Выделяем категориальные признаки
categorical_features = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

# Заполняем пропущенные значения в категориальных признаках самым часто встречающимся значением (модой)
dataset[categorical_features] = dataset[categorical_features].fillna(dataset[categorical_features].mode().iloc[0])


# Определяем функцию для нормализации числовых признаков
def normalize(df, columns):
    for col in columns:
        # Находим минимальное и максимальное значения в столбце
        min_val = df[col].min()
        max_val = df[col].max()
        # Если значения в столбце не одинаковые, то выполняем нормализацию по формуле (x - min) / (max - min)
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            # Если все значения в столбце одинаковые, то заменяем их на 0
            df[col] = 0.0
    return df


# Определяем функцию для перемешивания данных
def shuffle_data(dataframe, seed=42):
    # Устанавливаем фиксированное значение seed для воспроизводимости
    random.seed(seed)
    # Преобразуем данные в список списков для удобства перестановок
    lines = dataframe.values.tolist()
    # Алгоритм перемешивания: для каждого элемента выбираем случайный индекс и меняем элементы местами
    for i in range(len(lines) - 1, 0, -1):
        j = random.randint(0, i)
        lines[i], lines[j] = lines[j], lines[i]
    # Преобразуем данные обратно в DataFrame с исходными столбцами
    return pd.DataFrame(lines, columns=dataframe.columns)


# Нормализуем числовые признаки в датасете
dataset = normalize(dataset, numerical_features)

# Перемешиваем строки в датасете
dataset = shuffle_data(dataset, seed=42)

# Разбиваем датасет на обучающую и тестовую выборки (25% данных выделяется для теста)
train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)

# Сохраняем обработанные данные в отдельные CSV файлы
train_data.to_csv('../normalized_shuffled_train.csv', index=False)
test_data.to_csv('../normalized_shuffled_test.csv', index=False)
