import pandas as pd

input_file = '../datasets/dataset_modified.csv'
df = pd.read_csv(input_file)


df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 2})

# Переименование колонок (ракомментируй, если нужно)
# df.rename(columns={
#     'Calories Burn': 'Каллорий сожжено',
#     'Actual Weight': 'Вес',
#     'Age': 'Возраст',
#     'Gender': 'Пол',
#     'Duration': 'Продолжительность',
#     'Heart Rate': 'Частота сердцебиений',
#     'BMI': 'Индекс массы тела'
# }, inplace=True)

# Перемещение колонки 'Calories Burn' в конец
calories_column = df.pop('Calories Burn')
df['Calories Burn'] = calories_column


output_file = '../datasets/normalized_data.csv'
df.to_csv(output_file, index=False)
