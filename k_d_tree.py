import numpy as np

recursion_info = 0


# tree is build up of tuples in the shape of5 (left, vertex, right)
def build_tree(data, dim, depth=0):
    n = len(data)

    if not n:
        return None

    axis = depth % dim
    sort_axis_data = np.take(data, np.argsort(data[:, axis + 1], kind='stable', axis=0), axis=0)

    return build_tree(sort_axis_data[:n // 2], dim, depth + 1), \
           sort_axis_data[n // 2], \
           build_tree(sort_axis_data[n // 2 + 1:], dim, depth + 1)


# returns k-nearest neighbors in a given k-d tree relative to a point
def knn(data_tree, point, k, depth=0):
    global recursion_info

    recursion_info += 1

    if data_tree is None:  # empty children return nothing
        return [], None

    axis = depth % len(point)

    if point[axis] < data_tree[1][axis + 1]:
        next_branch = data_tree[0]
        opposite_branch = data_tree[2]
    else:
        next_branch = data_tree[2]
        opposite_branch = data_tree[0]

    best_k, dist = knn(next_branch, point, k, depth + 1)
    if len(best_k) == 0:
        best_k = np.empty((k, len(point) + 1))
        dist = np.ones(k) * np.inf

    new_dist = np.sum(np.square(data_tree[1][1:] - point))

    if np.max(dist) > new_dist:
        i = np.argmax(dist)
        best_k[i] = data_tree[1]
        dist[i] = new_dist

    if np.max(dist) > (point[axis] - data_tree[1][axis + 1]) ** 2:
        best_k2, dist2 = knn(opposite_branch, point, k, depth + 1)
        if not len(best_k2) == 0:
            dist = np.concatenate((dist, dist2), axis=0)
            indx = np.argpartition(dist, k)[:k]
            tmp = np.concatenate((best_k, best_k2))
            best_k = np.take(tmp, indx, axis=0)
            dist = np.take(dist, indx)

    return best_k, dist
