import matplotlib.pyplot as plt
from matplotlib.colors import *

def display_2d_dataset(data):
    plt.scatter(data[:,1], data[:,2], c=data[:, 0])

    plt.show()