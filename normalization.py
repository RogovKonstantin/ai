import pandas as pd

input_file = 'datasets/dataset_modified.csv'
df = pd.read_csv(input_file)

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 2})

# df.rename(columns={
#     'Calories Burn': 'Каллорий сожжено',
#     'Actual Weight': 'Вес',
#     'Age': 'Возраст',
#     'Gender': 'Пол',
#     'Duration': 'Продолжительность',
#     'Heart Rate': 'Частота сердцебиений',
#     'BMI': 'Индекс массы тела'
# }, inplace=True)

output_file = 'datasets/normalized_data.csv'
df.to_csv(output_file, index=False)
