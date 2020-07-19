import time
import numpy as np

import dataset
import visual
import k_d_tree
from gui import *

ARRAY_LIMIT = 50000     # lower if memory errors occur; splits the large arrays if they are too large
# equally important is using the 64-bit version of python

K = np.arange(50)


# returns failure rate of data2 compared to data1, parameters have to have same sorting and same shape
def R(data1, data2):
    n = len(data1)
    if not n == len(data2):
        return

    f = n - np.count_nonzero(np.isclose(data1, data2).all(axis=1))
    return f / n


# return list of indices of nearest points
# x is in array-shape
def k_nearest_brute_sort(data, x, k_max):
    x_split = np.array_split(x, len(x) * len(data) // ARRAY_LIMIT + 1)
    indx = []
    for x in x_split:
        # dist = sp.distance.cdist(data[:, 1:], x, 'sqeuclidean')  #  very fast but scipy library not allowed
        cen = np.repeat(data[:, 1:][:, np.newaxis, :], len(x), axis=1) - x  # (~0.6 sec)
        sq = np.square(cen)  # (~0.2 sec)
        dist = np.sum(sq, axis=2)
        part_indx = np.argpartition(dist, k_max, axis=0)[:k_max]
        sort_indx = np.argsort(np.take_along_axis(dist, part_indx, axis=0), axis=0)
        indx.append(np.take_along_axis(part_indx, sort_indx, axis=0))
    return np.concatenate(indx, axis=1)


def k_nearest_semi_sort(data, x, k):
    x_split = np.array_split(x, len(x) * len(data) // ARRAY_LIMIT + 1)
    indx = []
    for x in x_split:
        cen = np.repeat(data[:, 1:][:, np.newaxis, :], len(x), axis=1) - x
        sq = np.square(cen)
        dist = np.sum(sq, axis=2)
        indx.append(np.argpartition(dist, k, axis=0)[:k])
    return np.concatenate(indx, axis=1)


# computes f_D,k for given x values for k in array shape
def f_train_brute_sort(data, x, kset):
    near = k_nearest_brute_sort(data, x, np.max(kset))  # using k_nearest to only compute it once
    y = data[:, :1]
    nearest_bin = np.take_along_axis(y, near, axis=0)  # assembles array of nearest ys
    results = []
    for k in kset:
        result = np.sign(np.sum(nearest_bin[:k], axis=0))
        result[result == 0] = 1  # sets sign(0) to 1
        results.append(result)
    return results


# computes f_D,k for given x values
def f_test_semi_sort(data, x, k):
    near = k_nearest_semi_sort(data, x, k)
    y = data[:, :1]
    nearest_bin = np.take_along_axis(y, near, axis=0)
    result = np.sign(np.sum(nearest_bin, axis=0))
    result[result == 0] = 1  # sets sign(0) to 1
    return result


# computes f_D,k for given x values for k in array shape with k-d tree
def f_train_tree(data_tree, x, kset):
    results = np.zeros((len(kset), len(x)))
    for i, point in enumerate(x):
        near_data, dist = k_d_tree.knn(data_tree, point, np.max(kset))  # using k_nearest to only compute it once
        near_data = np.take(near_data, np.argsort(dist), axis=0)
        for j, k in enumerate(kset):
            result = np.sign(np.sum(near_data[:k, 0]))
            if result == 0:
                result = 1
            results[j, i] = result
    return results


# computes final f_D for given x values and k*
def f_final(data_segmented, x, k):
    tmp = np.zeros(len(x))

    for i, di in enumerate(data_segmented):
        di_complement = np.concatenate(np.delete(data_segmented, i, axis=0))
        tmp = tmp + f_test_semi_sort(di_complement, x, k)

    result = np.sign(tmp)
    result[result == 0] = 1  # sets sign(0) to 1
    return result


# stitches calculated y-values to the coordinates
def stitch(y, x):
    data = np.hstack((y[:, np.newaxis], x))
    return data


def classify(name, kset=K, l=5, output=True):
    dd, k_best = train_brute_sort(name, kset, l, output)
    print('k* =', k_best)

    test(dd, name, k_best, output)

    if False and output:  # only work in 2 dimensions
        grid(dd, k_best)


def grid(dd, k_best):
    grid_x = [[n / 100, m / 100] for n in range(100) for m in range(100)]
    visual.display_2d_dataset(stitch(f_final(dd, grid_x, k_best), grid_x), 'f evaluated to grid')  # Display grid


def test(dd, name, k_best,  output):
    test_data = dataset.parse('data/' + name + '.test.csv')
    if output:
        visual.display_2d_dataset(test_data, 'raw testing data')  # Display test data
    compare = f_final(dd, test_data[:, 1:], k_best)
    result_data = stitch(compare, test_data[:, 1:])
    f_rate = R(test_data, result_data)
    print('Failure rate (compared to test data):', f_rate)
    dataset.save_to_file('data/' + name + '.result.csv', result_data)
    if output:
        visual.display_2d_dataset(result_data, 'f evaluated to testing data')
        # Display result labels of test data

    return f_rate


def train_brute_sort(name, kset, l, output):
    train_data = dataset.parse('data/' + name + '.train.csv')
    if output:
        visual.display_2d_dataset(train_data, 'raw training data')  # Display training data

    # instead of making a random partition we use parts of a shuffled array
    # this results in disjoint sets d_i
    np.random.shuffle(train_data)
    # this way we have d_i = dd[i]
    dd = np.array_split(train_data, l)
    k_best_r = np.empty((l, len(kset)))
    for i, di in enumerate(dd):
        di_complement = np.concatenate(np.delete(dd, i, axis=0))

        for n, f in enumerate(f_train_brute_sort(di_complement, di[:, 1:], kset)):
            k_best_r[i][n] = R(di, stitch(f, di[:, 1:]))
    k_best = K[np.argmin(np.mean(k_best_r, axis=0))]
    return dd, k_best


def train_k_d_tree(name, kset, l, output):
    train_data = dataset.parse('data/' + name + '.train.csv')
    if output:
        visual.display_2d_dataset(train_data, 'raw training data')  # Display training data

    # instead of making a random partition we use parts of a shuffled array
    # this results in disjoint sets d_i
    np.random.shuffle(train_data)
    # this way we have d_i = dd[i]
    dd = np.array_split(train_data, l)
    d_trees = []
    k_best_r = np.empty((l, len(kset)))
    for i, di in enumerate(dd):
        di_complement = np.concatenate(np.delete(dd, i, axis=0))
        d_trees.append(k_d_tree.build_tree(di_complement, np.shape(di)[1] - 1))

    print('All trees constructed')

    for i, d_tree in enumerate(d_trees):
        for n, f in enumerate(f_train_tree(d_tree, dd[i][:, 1:], kset)):
            k_best_r[i][n] = R(dd[i], stitch(f, dd[i][:, 1:]))
    k_best = K[np.argmin(np.mean(k_best_r, axis=0))]
    return dd, k_best


def classify_all(kset=K, l=5):
    for data_file in dataset.datasets:
        print('Running dataset', data_file, '...')
        start_time = time.time()
        classify(data_file, kset, l, output=False)
        elapsed_time = time.time() - start_time
        print('Elapsed time:', elapsed_time, '\n')


#classify_all()

classify('bananas-1-2d', K, 5)

# debug statements

# d, k = train_k_d_tree('australian', K, 5, False)
# print(k)

gui = Gui()
gui.show()