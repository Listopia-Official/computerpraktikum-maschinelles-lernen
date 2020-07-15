import numpy as np
import scipy.spatial as sp  # Is the library scipy allowed?
import dataset
import time
import visual

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
def k_nearest(data, x):  # very time consuming
    dist = sp.distance.cdist(data[:, 1:], x, 'sqeuclidean')  # (~0.1 sec)
    return np.argsort(dist, axis=0)  # (~1.4 sec)


def k_nearest_alt(data, x, k):
    dist = sp.distance.cdist(data[:, 1:], x, 'sqeuclidean')  # (~0.1 sec)
    return np.argpartition(dist, k, axis=0)[:k]  # (~0.5 sec)


# computes f_D,k for given x value for k in array shape
def f_naive(data, x, kset):
    near = k_nearest(data, x)  # using k_nearest to only compute it once
    y = data[:, :1]
    nearest_bin = np.take_along_axis(np.repeat(y, len(x), axis=1), near, axis=0)
    results = []
    for k in kset:
        result = np.sign(np.sum(nearest_bin[:k], axis=0))
        result[result == 0] = 1  # sets sign(0) to 1
        results.append(result)
    return results


# computes f_D,k for given x values
def f_naive_alt(data, x, k):
    near = k_nearest_alt(data, x, k)  # using k_nearest_alt because it is faster
    y = data[:, :1]
    nearest_bin = np.take_along_axis(np.repeat(y, len(x), axis=1), near, axis=0)
    result = np.sign(np.sum(nearest_bin, axis=0))
    result[result == 0] = 1  # sets sign(0) to 1
    return result


# computes final f_D for given x values and k*
def f_final(data_segmented, x, k):
    tmp = np.zeros(len(x))

    for i, di in enumerate(data_segmented):
        di_complement = np.concatenate(np.delete(data_segmented, i, axis=0))
        tmp = tmp + f_naive_alt(di_complement, x, k)

    result = np.sign(tmp)
    result[result == 0] = 1  # sets sign(0) to 1
    return result


# stitches calculated y-values to the coordinates
def stitch(y, x):
    data = np.hstack((y[:, np.newaxis], x))
    return data


def classify(name, kset=K, l=5, output=True):
    train = dataset.parse('data/' + name + '.train.csv')

    if output:
        visual.display_2d_dataset(train)  # Display training

    # instead of making a random partition we use parts of a shuffled array
    # this results in disjoint sets d_i (what would arbitrary sets imply?)
    np.random.shuffle(train)
    # this way we have d_i = dd[i]
    dd = np.array_split(train, l)

    k_best_r = np.ones((l, len(kset)))

    for i, di in enumerate(dd):
        di_complement = np.concatenate(np.delete(dd, i, axis=0))

        for n, f in enumerate(f_naive(di_complement, di[:, 1:], kset)):
            k_best_r[i][n] = R(di, stitch(f, di[:, 1:]))

    k_best = K[np.argmin(np.mean(k_best_r, axis=0))]
    print('k* =', k_best)

    test = dataset.parse('data/' + name + '.test.csv')

    if output:
        visual.display_2d_dataset(test)  # Display test data

    compare = f_final(dd, test[:, 1:], k_best)
    print('Failure rate (compared to test data):', R(test, stitch(compare, test[:, 1:])))
    if output:
        visual.display_2d_dataset(stitch(compare, test[:, 1:]))  # Display guessed labels of test data

    # grid = [[n/100, m/100] for n in range(100) for m in range(100)]
    # visual.display_2d_dataset(stitch(f_final(dd, grid, k_best), grid)) # Display grid


def classify_all(kset=K, l=5):
    for data_file in dataset.datasets:
        print('Running dataset', data_file, '...')
        start_time = time.time()
        classify(data_file, kset, l, output=False)
        elapsed_time = time.time() - start_time
        print('Elapsed time:', elapsed_time, '\n')


# classify_all()

classify('bananas-1-2d', K, 5)

# debug statements
# timeit.timeit("main.classify('bananas-1-4d')", "import main", number=1)


# print(timeit.timeit("numpy.argsort(dist, axis=0)[:20]",
#                    "import dataset, scipy.spatial, numpy\n" +
#                    "dat = dataset.parse('data/toy-2d.train.csv')\n" +
#                    "dist = scipy.spatial.distance.cdist(dat[:, 1:], dat[:, 1:], 'sqeuclidean')", number=10))

# dat = dataset.parse('data_artificial/toy-2d.train.csv')

# print("prev", k_nearest(dat, np.array([[1, 2], [0, 0], [1, 0]]), 10))
# print("now", k_nearest_alt(dat, np.array([[1, 2], [0, 0], [1, 0]]), 10))
# print(f_naive(dat, dat[:, 1:], 200))
# print(R(stitch(f_naive(dat, dat[:, 1:], 15), dat[:, 1:]), dat)) # compares naive f to training data
