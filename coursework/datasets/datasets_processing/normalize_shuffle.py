import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Определяем функцию для перемешивания данных
def shuffle_data(dataframe, seed=42):
    """
    Перемешивает строки DataFrame вручную для воспроизводимости результатов.
    """
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

def preprocess_and_save():
    """
    1) Загружает ../dataset.csv с использованием pandas.
    2) Удаляет строки с пропущенными значениями.
    3) Применяет стандартизацию Z-score к числовым признакам.
    4) Сохраняет моду категориальных признаков.
    5) Перемешивает набор данных.
    6) Делит данные на обучающую и тестовую выборки и сохраняет их в файлы:
       - ../normalized_shuffled_train.csv
       - ../normalized_shuffled_test.csv
    7) Сохраняет график сравнения распределений в ../standardization_comparison_all_features.png.
    """

    # ------------------ Загрузка набора данных ------------------ #
    original_dataset = pd.read_csv('../dataset.csv')

    # Удаление строк с любыми пропущенными значениями
    original_dataset = original_dataset.dropna()

    # Определение числовых и категориальных признаков
    numerical_features = [
        'age', 'cigsPerDay', 'totChol', 'sysBP',
        'diaBP', 'BMI', 'heartRate', 'glucose'
    ]
    categorical_features = [
        'male', 'education', 'currentSmoker', 'BPMeds',
        'prevalentStroke', 'prevalentHyp', 'diabetes'
    ]

    # Создаем копию данных для стандартизации
    dataset = original_dataset.copy()

    # --------------- Применяем стандартизацию Z-score --------------- #
    means = {}
    scales = {}
    for feature in numerical_features:
        mean = dataset[feature].mean()
        std = dataset[feature].std()
        means[feature] = mean
        scales[feature] = std
        dataset[feature] = (dataset[feature] - mean) / std

    # --------------- Сохраняем моду категориальных признаков --------------- #
    modes = {}
    for feature in categorical_features:
        modes[feature] = dataset[feature].mode()[0]

    # Сохраняем параметры масштабирования и моды
    print("Сохранение параметров масштабирования и мод в файл:", 'scaler_params.pkl')
    with open('scaler_params.pkl', "wb") as f:
        pickle.dump({"means": means, "scales": scales, "modes": modes}, f)

    # ------------------ Перемешивание набора данных ------------------ #
    dataset = shuffle_data(dataset, seed=42)

    # --------------- Разделение набора данных на обучающую и тестовую выборки --------------- #
    train_size = int(0.75 * len(dataset))
    train_data = dataset.iloc[:train_size]
    test_data = dataset.iloc[train_size:].drop(columns=['TenYearCHD'])

    # --------------- Сохранение обработанных данных --------------- #
    train_data.to_csv('../normalized_shuffled_train.csv', index=False)
    test_data.to_csv('../normalized_shuffled_test.csv', index=False)

    # --------------- Визуализация: сравнение распределений --------------- #
    fig, axes = plt.subplots(len(numerical_features), 1, figsize=(10, 5 * len(numerical_features)))

    for idx, feature in enumerate(numerical_features):
        axes[idx].hist(original_dataset[feature], bins=30, alpha=0.5, label='До стандартизации', color='blue')
        axes[idx].hist(dataset[feature], bins=30, alpha=0.5, label='После стандартизации', color='orange')
        axes[idx].set_title(f'Распределение признака "{feature}"')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Частота')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig('../standardization_comparison_all_features.png')
    plt.close()


# Добавляем основной блок для выполнения функции
if __name__ == "__main__":
    preprocess_and_save()
