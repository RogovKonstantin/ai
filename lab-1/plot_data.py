import matplotlib.pyplot as plt


def plot_data(x, y):
    plt.scatter(x, y, marker='x', color='green')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль')
    plt.title('График зависимости прибыли от количества автомобилей')
    plt.savefig("result.png")
