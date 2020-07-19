import matplotlib.pyplot as plt

"""
A module used to visualize the data used by this application via matplotlib.
The module is intended for 2D-data, however it can also visualize the first two axis of data of higher-dimensions.
"""

"""
Input:
 data: The data in the format as specified in dataset.py
 title: The plit title
 micro: Whether the plot is rendered with a smaller size
"""
def display_2d_dataset(data, title='scatter plot', micro=False):

    size = (7, 7) # Normal size

    if micro:
        size = (3, 3) # Small size

    fig = plt.figure(title, figsize=size)
    plt.clf() # Clear figure
    plt.scatter(data[:, 1], data[:, 2], s=1.4, c=data[:, 0], cmap='bwr') # Plot the points and set the colors
    return fig
