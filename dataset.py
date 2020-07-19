import numpy as np

"""
This is the (small) module for handling importing and exporting the datasets.
"""

"""
Parses the file at the specified path and returns the data in the following format:
 - It's an array of shape (n, 1 + dim)
 - y-values are accessible with d[:,:1] (2D-case, analogous for higher dimensions)
 - x-values are accessible with d[:,1:] (2D-case)
"""
def parse(path):
    return np.genfromtxt(path, delimiter=',')

# Saves the specified dataset to the specified path
def save_to_file(path, data):
    np.savetxt(path, data, fmt='%g', delimiter=', ')
