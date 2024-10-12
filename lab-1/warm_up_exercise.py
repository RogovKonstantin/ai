import numpy as np


def warmup_exercise_built_in(n):
    matrix = np.eye(n).astype(int)
    print(f'Единичная матрица с использованием numpy: \n{matrix}')


def warmup_exercise_manual(n):
    matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    # print(f'Единичная матрица без использованием numpy: \n{matrix}')
    for row in matrix:
        print(row)
