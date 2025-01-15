"""
Файл: gui_predict.py

Простое графическое приложение на Tkinter, которое:
1) Запрашивает 15 признаков из вашего набора данных:
   - 8 числовых признаков, требующих стандартизации
   - 7 категориальных признаков (например, 0 или 1).
2) Применяет ту же стандартизацию (используя сохраненные средние и стандартные отклонения).
3) Загружает веса логистической регрессии из файла model_manual_weights.txt.
4) Показывает предсказанный результат.
"""

import tkinter as tk
import pickle
import math

# 1) Определяем числовые и категориальные признаки с переводом.
numerical_features = {
    'age': 'Возраст',
    'cigsPerDay': 'Сигарет в день',
    'totChol': 'Общий холестерин',
    'sysBP': 'Систолическое АД',
    'diaBP': 'Диастолическое АД',
    'BMI': 'Индекс массы тела',
    'heartRate': 'Частота пульса',
    'glucose': 'Уровень глюкозы'
}
categorical_features = {
    'male': 'Мужчина',
    'education': 'Уровень образования',
    'currentSmoker': 'Курильщик',
    'BPMeds': 'Препараты от давления',
    'prevalentStroke': 'Инсульт в анамнезе',
    'prevalentHyp': 'Гипертония',
    'diabetes': 'Диабет'
}
all_features = list(numerical_features.keys()) + list(categorical_features.keys())

# 2) Загружаем параметры масштабирования.
try:
    scaler_params = pickle.load(open("../datasets/datasets_processing/scaler_params.pkl", "rb"))
    means = scaler_params["means"]
    scales = scaler_params["scales"]
    modes = scaler_params["modes"]  # Добавляем моды категориальных признаков
except Exception as e:
    print("Ошибка загрузки параметров масштабирования:", e)
    means = {}
    scales = {}
    modes = {}

# 3) Функция для загрузки весов.
def load_manual_weights(file_path):
    with open(file_path, 'r') as f:
        weights_str = f.read().strip().split(',')
    return list(map(float, weights_str))

# 4) Сигмоида.
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# 5) Предсказание.
def predict_manual(inputs, weights):
    x_with_bias = [1.0] + inputs
    z = sum(w * x for w, x in zip(weights, x_with_bias))
    return 1 if sigmoid(z) >= 0.5 else 0

# 6) Графический интерфейс.
class HeartDiseasePredictorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Предсказание")

        self.entries = {}
        row_index = 0

        # Числовые признаки
        tk.Label(master, text="Введите числовые признаки:").grid(row=row_index, column=0, columnspan=2, pady=5, sticky="w")
        row_index += 1
        for feat, label in numerical_features.items():
            tk.Label(master, text=f"{label}:").grid(row=row_index, column=0, padx=5, pady=2, sticky="e")
            entry = tk.Entry(master)
            entry.insert(0, f"{means.get(feat, 0):.2f}")
            entry.grid(row=row_index, column=1, padx=5, pady=2)
            self.entries[feat] = entry
            row_index += 1

        # Категориальные признаки
        tk.Label(master, text="Введите категориальные признаки:").grid(row=row_index, column=0, columnspan=2, pady=5, sticky="w")
        row_index += 1
        for feat, label in categorical_features.items():
            tk.Label(master, text=f"{label}:").grid(row=row_index, column=0, padx=5, pady=2, sticky="e")
            entry = tk.Entry(master)
            entry.insert(0, f"{modes.get(feat, 0)}")  # Используем моду
            entry.grid(row=row_index, column=1, padx=5, pady=2)
            self.entries[feat] = entry
            row_index += 1

        # Кнопка предсказания
        self.predict_button = tk.Button(master, text="Предсказать", command=self.on_predict)
        self.predict_button.grid(row=row_index, column=0, columnspan=2, pady=10)

        row_index += 1
        self.result_label = tk.Label(master, text="", font=("Helvetica", 12, "bold"))
        self.result_label.grid(row=row_index, column=0, columnspan=2, pady=10)

        self.manual_weights = load_manual_weights("../models/model_manual_weights.txt")

    def on_predict(self):
        # Проверяем ввод
        for key, entry in self.entries.items():
            if not entry.get().strip():
                self.result_label.config(text="Заполните все поля!", fg="red")
                return

        # Считываем ввод
        user_input_numeric = [float(self.entries[feat].get()) for feat in numerical_features]
        user_input_categorical = [float(self.entries[feat].get()) for feat in categorical_features]

        # Стандартизируем числовые признаки
        standardized_numeric = [(val - means.get(feat, 0)) / scales.get(feat, 1) for val, feat in zip(user_input_numeric, numerical_features)]

        all_inputs = standardized_numeric + user_input_categorical

        # Предсказание
        prediction = predict_manual(all_inputs, self.manual_weights)

        msg = "Результат: риск есть (1)" if prediction == 1 else "Результат: риска нет (0)"
        self.result_label.config(text=msg, fg="green")


if __name__ == "__main__":
    root = tk.Tk()
    gui = HeartDiseasePredictorGUI(root)
    root.mainloop()
