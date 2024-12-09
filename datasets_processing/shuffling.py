import random


def read_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def write_csv(file_path, lines):
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def shuffle_data(lines):
    for _ in range(1000):
        for i in range(len(lines) - 1, 0, -1):
            j = random.randint(0, i)
            lines[i], lines[j] = lines[j], lines[i]


def main(input_file, output_file):
    lines = read_csv(input_file)

    header = lines[0]
    data_lines = lines[1:]
    shuffle_data(data_lines)

    shuffled_lines = [header] + data_lines

    write_csv(output_file, shuffled_lines)


input_file_path = '../dataset/dataset_cleaned.csv'
output_file_path = '../dataset/dataset_final.csv'

main(input_file_path, output_file_path)
