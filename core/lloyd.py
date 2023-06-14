# -*- coding = utf-8 -*-
# lloyd algorithm to use
# 2023.03.02,edit by Lin

import numpy as np
import math
from scipy.linalg.blas import sgemm


def row_norm(X):
    """
    Row-wise (squared) Euclidean norm of X
    """
    norms = np.einsum("ij,ij->i", X, X)
    return norms


def euclidean_dense_dense(a, b, squared=True):
    """
    Euclidean distance between a dense and b dense.
    """
    n_features = a.shape[0]
    result = 0
    for i in range(n_features):
        result += (a[i] - b[i]) ** 2
    return result if squared else math.sqrt(result)


def update_dense(X, centers_old, centers_squared_norms, update_centers=True):
    """
    Compute the partial contribution of a single data to the labels and centers.
    """
    n_samples = X.shape[0]
    n_clusters = centers_old.shape[0]
    pairwise_distances = np.zeros([n_samples, n_clusters], dtype=X.dtype)
    labels = np.full(n_samples, -1, dtype=np.int32)
    weight_in_clusters = np.zeros(n_clusters, dtype=np.int32)
    centers_new = np.zeros_like(centers_old)

    for i in range(n_samples):
        for j in range(n_clusters):
            pairwise_distances[i][j] = centers_squared_norms[j]
    pd = sgemm(-2, X, centers_old, trans_b=1, beta=1, c=pairwise_distances)

    for i in range(n_samples):
        min_sq_dist = pd[i, 0]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pd[i, j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += 1
            for k in range(X.shape[1]):
                centers_new[label, k] += X[i, k]

    return labels, centers_new, weight_in_clusters


def relocate_empty_clusters_dense(X, centers_old, centers_new, weight_in_clusters, labels):
    """
    Relocate centers which have no sample assigned to them.
    """
    empty_clusters = np.where(np.equal(weight_in_clusters, 0))[0].astype(np.int32)
    n_empty = empty_clusters.shape[0]

    if n_empty == 0:
        return centers_new, weight_in_clusters

    n_features = X.shape[1]
    distances = ((np.asarray(X) - np.asarray(centers_old)[labels]) ** 2).sum(axis=1)
    far_from_centers = np.argpartition(distances, -n_empty)[:-n_empty - 1:-1].astype(np.int32)

    for idx in range(n_empty):
        new_cluster_id = empty_clusters[idx]

        far_idx = far_from_centers[idx]
        weight = 1

        old_cluster_id = labels[far_idx]

        for k in range(n_features):
            centers_new[old_cluster_id, k] -= X[far_idx, k] * weight
            centers_new[new_cluster_id, k] = X[far_idx, k] * weight

        weight_in_clusters[new_cluster_id] = weight
        weight_in_clusters[old_cluster_id] -= weight

    return centers_new, weight_in_clusters


def average_centers(centers, weight_in_clusters):
    """
    Average new centers by weights.
    """
    n_clusters = centers.shape[0]
    n_features = centers.shape[1]
    for j in range(n_clusters):
        if weight_in_clusters[j] > 0:
            alpha = 1.0 / weight_in_clusters[j]
            for k in range(n_features):
                centers[j, k] *= alpha
    return centers


def center_shift(centers_old, centers_new):
    """
    Compute shift between old and new centers.
    """
    n_clusters = centers_old.shape[0]
    centers_shift = np.zeros(n_clusters, dtype=centers_old.dtype)
    for j in range(n_clusters):
        centers_shift[j] = euclidean_dense_dense(centers_new[j], centers_old[j], squared=False)

    return centers_shift


def lloyd_iter_dense(X, centers_old, update_centers=True):
    """
    Single iteration of K-means lloyd algorithm.
    :param X: Input data
    :param centers_old: Centers before previous iteration
    :param update_centers: bool, default=True
                - True: The labels and the new centers will be computed.
                - False: Only the labels will be computed.
    """
    centers_squared_norms = row_norm(centers_old)
    labels, centers_new, weight_in_clusters = update_dense(X, centers_old, centers_squared_norms,
                                                           update_centers=update_centers)

    if update_centers:
        centers_new, weight_in_clusters = relocate_empty_clusters_dense(X, centers_old, centers_new, weight_in_clusters,
                                                                        labels)
        centers_new = average_centers(centers_new, weight_in_clusters)
    centers_shift = center_shift(centers_old, centers_new)

    return centers_new, weight_in_clusters, labels, centers_shift


def inertia_dense(X, centers, labels):
    """
    Sum of squared distance between each sample and its assigned center.
    """
    single_label = -1
    n_samples = X.shape[0]
    inertia = 0.0

    for i in range(n_samples):
        j = labels[i]
        if single_label < 0 or single_label == j:
            sq_dist = euclidean_dense_dense(X[i], centers[j], squared=True)
            inertia += sq_dist

    return inertia


def is_same_clustering(labels1, labels2, n_clusters):
    """
    Check if two arrays of labels are the same up to a permutation of the labels.
    """
    mapping = np.full(fill_value=-1, shape=(n_clusters,), dtype=np.int32)
    for i in range(labels1.shape[0]):
        if mapping[labels1[i]] == -1:
            mapping[labels1[i]] = labels2[i]
        elif mapping[labels1[i]] != labels2[i]:
            return False

    return True


def kmeans_single_lloyd(X, centers_init, max_iter=100, tol=1e-4):
    """
    A single run of k-means lloyd.
    :param X: Input data
    :param centers_init: The initial centers
    :param max_iter: Maximum number of iterations of the k-means algorithm for a single run.
    :param tol: The difference in the cluster centers of two consecutive iterations to declare convergence.
    """
    centers = centers_init
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    strict_convergence = False

    for i in range(max_iter):
        centers_new, weight_in_clusters, labels, centers_shift = lloyd_iter_dense(X, centers, update_centers=True)
        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            strict_convergence = True
            break
        else:
            center_shift_tot = (centers_shift ** 2).sum()
            if center_shift_tot <= tol:
                break

        labels_old[:] = labels

    if not strict_convergence:
        centers, weight_in_clusters, labels, centers_shift = lloyd_iter_dense(X, centers, update_centers=False)

    inertia = inertia_dense(X, centers, labels)

    return labels, inertia, centers, i + 1
