# -*- coding = utf-8 -*-
# Kmeans++ method
# 2023.02.27,edit by Lin

import numpy as np
from .lloyd import row_norm, kmeans_single_lloyd, is_same_clustering


def euclidean_distances(X, Y=None, X_norm_squared=None, Y_norm_squared=None):
    """
    Compute the euclidean distances.
    """
    if X_norm_squared is not None:
        if X_norm_squared.dtype == np.float32:
            XX = None
        else:
            XX = X_norm_squared.reshape(-1, 1)
    elif X.dtype == np.float32:
        XX = None
    else:
        XX = row_norm(X)[:, np.newaxis]

    if Y is X:
        YY = None if XX is None else XX.T
    else:
        if Y_norm_squared is not None:
            if Y_norm_squared.dtype == np.float32:
                YY = None
            else:
                YY = Y_norm_squared.reshape(1, -1)
        elif Y.dtype == np.float32:
            YY = None
        else:
            YY = row_norm(Y)[np.newaxis, :]

    distances = -2 * (X @ Y.T)
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances


def kmeans_plusplus(X, k, x_squared_norms, random_state):
    """
    Computational component for initialization of k by k-means++.
    :param X: Input data
    :param k: The number of clusters
    :param x_squared_norms: Squared euclidean norm of each data point
    :param random_state: Determines random number generation for centroid initialization
    """
    n_samples, n_features = X.shape
    centers = np.empty((k, n_features), dtype=X.dtype)
    n_local_trials = 2 + int(np.log(k))

    center_id = random_state.randint(n_samples)
    indices = np.full(k, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
    )
    current_pot = closest_dist_sq.sum()
    for c in range(1, k):
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        stable_cumsum = np.cumsum(closest_dist_sq, axis=None, dtype=np.float64)
        candidate_ids = np.searchsorted(stable_cumsum, rand_vals)
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms,
        )

        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return centers, indices


def centroids(X, k, x_squared_norms, random_state):
    """
    Compute the initial centroids.
    """
    n_clusters = k
    centers, _ = kmeans_plusplus(X, n_clusters, random_state=random_state, x_squared_norms=x_squared_norms)

    return centers


def kmeans(X, k, n_init=10, max_iter=300):
    """
    Main function for K-Means clustering.
    :param X: Input data
    :param k: The number of clusters
    :param n_init: Number of time the k-means algorithm will be run with different centroid seeds.
    :param max_iter: Maximum number of iterations of the k-means algorithm for a single run.
    """
    random_state = np.random.mtrand._rand

    X_mean = X.mean(axis=0)
    X -= X_mean
    x_squared_norms = row_norm(X)
    best_inertia, best_labels = None, None

    for i in range(n_init):
        centers_init = centroids(X, k, x_squared_norms=x_squared_norms, random_state=random_state)
        labels, inertia, centers, n_iter_ = kmeans_single_lloyd(X, centers_init, max_iter=max_iter)
        if best_inertia is None or (inertia < best_inertia and not is_same_clustering(labels, best_labels, k)):
            best_labels = labels
            best_inertia = inertia

    return best_labels
