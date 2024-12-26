import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        # Если значения в столбце не одинаковые, то выполняем нормализацию по формуле (x - min) / (max - min) [min-max нормализация]
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

# Создаем копию датасета для сравнения до и после нормализации
original_dataset = dataset.copy()

# Нормализуем числовые признаки в датасете
dataset = normalize(dataset, numerical_features)

# Перемешиваем строки в датасете
dataset = shuffle_data(dataset, seed=42)

# Разбиваем датасет на обучающую и тестовую выборки (25% данных выделяется для теста)
train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)

# Сохраняем обработанные данные в отдельные CSV файлы
train_data.to_csv('../normalized_shuffled_train.csv', index=False)
test_data.to_csv('../normalized_shuffled_test.csv', index=False)

# Создаем сетку для визуализации всех признаков в одном файле
rows = (len(numerical_features) + 2) // 3  # Определяем количество строк для сетки
fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))  # Сетка 3 графика в строке
axes = axes.flatten()  # Преобразуем массив осей для удобства индексации

# Визуализация сравнения для каждого признака
for idx, feature in enumerate(numerical_features):
    sns.histplot(original_dataset[feature], bins=30, kde=True, color='blue', label='До нормализации',
                 stat="density", alpha=0.5, ax=axes[idx])
    sns.histplot(dataset[feature], bins=30, kde=True, color='orange', label='После нормализации',
                 stat="density", alpha=0.5, ax=axes[idx])
    axes[idx].set_title(f'Распределение "{feature}"')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Плотность')
    axes[idx].legend()

# Удаляем пустые подграфики, если число признаков не кратно 3
for ax in axes[len(numerical_features):]:
    fig.delaxes(ax)

# Устанавливаем общий заголовок и сохраняем файл
fig.suptitle('Сравнение распределений до и после нормализации', fontsize=16, y=0.92)
plt.tight_layout()
plt.savefig('../normalization_comparison_all_features.png')
plt.close()

# Дополнительная визуализация: "до" и "после" в виде точек на одном графике
fig, ax = plt.subplots(figsize=(10, 6))
for feature in numerical_features:
    ax.scatter(original_dataset[feature], dataset[feature], alpha=0.5, label=feature)
ax.set_title('Сравнение значений до и после нормализации')
ax.set_xlabel('Значения до нормализации')
ax.set_ylabel('Значения после нормализации')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('../normalization_scatter_comparison.png')
plt.close()
