import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

def display_2d_dataset(data, title='scatter plot'):
    # plt.ion()
    fig = plt.figure(title)
    plt.clf()
    plt.scatter(data[:, 1], data[:, 2], s=1.4, c=data[:, 0], cmap='bwr')
    return fig
