import pandas as pd

# Замените 'input.csv' на путь к вашему исходному CSV-файлу
input_csv_path = '../dataset.csv'
# Замените 'output.xlsx' на желаемое название/путь для XLSX-файла
output_xlsx_path = '../output.xlsx'

# Словарь для переименования столбцов
column_map = {
    'male': 'Пол (0=жен., 1=муж.)',
    'age': 'Возраст (лет)',
    'education': 'Уровень образования (код)',
    'currentSmoker': 'Курит сейчас (да=1/нет=0)',
    'cigsPerDay': 'Сигарет в день (шт.)',
    'BPMeds': 'Принимает лекарства от давления (да=1/нет=0)',
    'prevalentStroke': 'Инсульт (да=1/нет=0)',
    'prevalentHyp': 'Гипертензия (да=1/нет=0)',
    'diabetes': 'Диабет (да=1/нет=0)',
    'totChol': 'Общий холестерин (мг/дл)',
    'sysBP': 'Систолическое АД (мм рт. ст.)',
    'diaBP': 'Диастолическое АД (мм рт. ст.)',
    'BMI': 'Индекс массы тела (кг/м²)',
    'heartRate': 'ЧСС (уд./мин)',
    'glucose': 'Глюкоза (мг/дл)',
    'TenYearCHD': 'Риск ИБС на 10 лет (да=1/нет=0)'
}

def convert_csv_to_xlsx(input_file, output_file, rename_dict):
    # Считываем исходный CSV
    df = pd.read_csv(input_file)

    # Переименовываем столбцы
    df.rename(columns=rename_dict, inplace=True)

    # Сохраняем в XLSX
    df.to_excel(output_file, index=False)
    print(f"Файл успешно сохранён как '{output_file}'.")

# Запуск функции
convert_csv_to_xlsx(input_csv_path, output_xlsx_path, column_map)
