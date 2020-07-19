import numpy as np

datasets = ['australian', 'bananas-1-2d', 'bananas-1-4d', 'bananas-2-2d', 'bananas-2-4d',
            'bananas-5-2d', 'bananas-5-4d', 'cod-rna.5000', 'crosses-2d', 'ijcnn1.5000', 'ijcnn1.10000', 'ijcnn1',
            'svmguide1', 'toy-2d', 'toy-3d', 'toy-4d', 'toy-10d']


# generates array of shape (n, 1 + dim)
# y-values are accessible with d[:,:1]
# x-values are accessible with d[:,1:]
def parse(path):
    d = np.genfromtxt(path, delimiter=',')

    return d


def save_to_file(path):
    pass  # TODO save functionality
