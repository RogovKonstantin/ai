import numpy as np

col1 = np.array([[1],
                 [2],
                 [3]])
col2 = np.array([[4],
                 [5],
                 [6]])

print(col1.shape)
print(col2.shape)

vectorized = np.sum(col1 * col2)
print(f"Скалярное произведение (сумма произведений): {vectorized}")

byElements = 0
for i in range(len(col1)):
    byElements += col1[i][0] * col2[i][0]
print(f"Скалярное произведение (поэлементное произведение): {byElements}")

dot_product_vectorized = np.dot(col1.T, col2).item()
print(f"Скалярное произведение (векторный подход): {dot_product_vectorized}")
