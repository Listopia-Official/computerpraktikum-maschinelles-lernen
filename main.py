import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # only used as a comparable implementation

import dataset
import visual
import k_d_tree
from gui import *

ARRAY_LIMIT = 50000     # lower if memory errors occur; splits the large arrays if they are too large
# equally important is using the 64-bit version of python

K = np.arange(1, 201)


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


# returns list of k nearest points (unsorted)
# x is in array-shape
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


# computes f_D,k for given x values for one k
def f_test_semi_sort(data, x, k):
    near = k_nearest_semi_sort(data, x, k)
    y = data[:, :1]
    nearest_bin = np.take_along_axis(y, near, axis=0)
    result = np.sign(np.sum(nearest_bin, axis=0))
    result[result == 0] = 1  # sets sign(0) to 1
    return result


# computes f_D,k for given x values with k in array shape with k-d tree
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


# computes f_D,k for given x values for k in array shape with k-d tree
def f_test_tree(data_tree, x, k):
    near_data, dist = k_d_tree.knn(data_tree, x, k)

    result = np.sign(np.sum(near_data[:, 0]))
    if result == 0:
        result = 1
    return result


# computes final f_D for given x values and k* (only brute-sort)
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


def classify_gui(train_data, test_data, output_path, kset=K, l=5, algorithm='brute-sort'):
    if algorithm == 'brute_sort':
        dd, k_best = train_brute_sort(train_data, kset, l)
        print('k* =', k_best)
        f_rate, result_data = test(dd, test_data, k_best, output_path)
        return k_best, f_rate, result_data, dd
    elif algorithm == 'k-d_tree':
        dd, k_best = train_k_d_tree(train_data, kset, l)
        print('k* =', k_best)
        f_rate, result_data = test_k_d_tree(dd, test_data, k_best, output_path)
        return k_best, f_rate, result_data, dd
    elif algorithm == 'sklearn':  # Comparing with existing implementation
        sk_classifier = KNeighborsClassifier(n_neighbors=np.max(K))
        bool_y = np.copy(train_data[:, 0])
        bool_y[bool_y == -1] = 0
        sk_classifier.fit(train_data[:, 1:], bool_y)
        result = sk_classifier.predict(test_data[:, 1:])
        result[result == 0] = -1
        result_data = stitch(result, test_data[:, 1:])
        f_rate = R(result_data, test_data)
        print("k* = unused")
        print('Failure rate (compared to test data):', f_rate)
        return np.NAN, f_rate, result_data


def grid(dd, k_best, grid_size):
    grid_x = [[n / grid_size, m / grid_size] for n in range(grid_size) for m in range(grid_size)]
    visual.display_2d_dataset(stitch(f_final(dd, grid_x, k_best), grid_x), 'f evaluated to grid')  # Display grid


# compares prediction based on k* with test data and saves result_data
def test(dd, test_data, k_best,  output_path):
    compare = f_final(dd, test_data[:, 1:], k_best)
    result_data = stitch(compare, test_data[:, 1:])
    f_rate = R(test_data, result_data)
    print('Failure rate (compared to test data):', f_rate)
    dataset.save_to_file(output_path, result_data)

    return f_rate, result_data


def train_brute_sort(train_data, kset, l):
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
    k_best = kset[np.argmin(np.mean(k_best_r, axis=0))]
    return dd, k_best


def train_k_d_tree(train_data, kset, l):
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
    k_best = kset[np.argmin(np.mean(k_best_r, axis=0))]
    return d_trees, k_best


def test_k_d_tree(d_trees, test_data, k_best, output_path):
    compare = np.zeros(len(test_data))
    for n, x in enumerate(test_data):
        for i, d_tree in enumerate(d_trees):
            compare[n] += f_test_tree(d_tree, x[1:], k_best)
    compare = np.sign(compare)
    compare[compare == 0] = 1
    result_data = stitch(compare, test_data[:, 1:])
    f_rate = R(test_data, result_data)
    print('Failure rate (compared to test data):', f_rate)
    dataset.save_to_file(output_path, result_data)

    return f_rate, result_data


gui = Gui(classify_gui, grid)
gui.show()

# classify(dataset.parse('data/bananas-1-2d.train.csv'), dataset.parse('data/bananas-1-2d.test.csv'), 'data/')
