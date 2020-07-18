import matplotlib.pyplot as plt
from matplotlib.colors import Colormap


def display_2d_dataset(data, title='scatter plot'):
    plt.figure(title)
    plt.scatter(data[:, 1], data[:, 2], s=1.4, c=data[:, 0], cmap='bwr')

    plt.show()
