# -*- coding = utf-8 -*-
# Create a similarity matrix.
# 2022.08.10,edit by Lin

import math
import numpy as np
from .ck_dist import ckdist


def pdist(X, k):
    """
    Pairwise distance between sets of X and X.
    :param X: Input data
    :param k: Number of smallest distances to find
    :return: A matrix D containing the ckdist distances between each pair of observations
    """
    nx = len(X)
    p = len(X[0])

    # Check the input data
    for i in range(nx):
        for j in range(p):
            if isinstance(X[i][j], float) and not math.isnan(X[i][j]):
                continue
            else:
                raise ValueError('The input data are wrong!')

    X = np.array(X)
    smallest_flag = min(k, nx)
    D = np.zeros((smallest_flag, nx), dtype=np.float64)
    I = np.zeros((smallest_flag, nx), dtype=np.int32)

    for i in range(nx):
        temp = ckdist(X[i], X)
        D[:, i], I[:, i] = partial_sort(temp, smallest_flag)

    return D, I


def partial_sort(D, smallest):
    """
    Sort D constraint by smallest
    """
    I = np.argsort(D)
    D = np.sort(D)
    D = D[:smallest]
    I = I[:smallest]

    return D, I


def knn_search(X, numNN):
    """
    Find K nearest neighbors using an ExhaustiveSearcher object.
    :param X: Input data
    :param numNN: Neighbors numbers
    """
    nx = len(X)
    nd = len(X[0])

    if numNN < 3:
        extra = 2
    elif numNN < 5:
        extra = numNN
    else:
        extra = 5
    K = numNN + extra

    dist2, idx2 = pdist(X, K)
    dist2 = dist2.T
    idx2 = idx2.T
    idx = [[] for _ in range(nx)]
    dist = [[] for _ in range(nx)]
    doneidx = []
    notdoneidx = []

    if numNN >= nx:
        for i in range(nx):
            idx[i] = idx2[i, :].tolist()
            dist[i] = dist2[i, :].tolist()
        idx = np.array(idx)
        dist = np.array(dist)
    else:
        for i in range(nx):
            if dist2[i][numNN + 1] > dist2[i][numNN] or math.isnan(dist2[i][numNN + 1]):
                doneidx.append(i)
            else:
                notdoneidx.append(i)
        donelen = len(doneidx)
        for i in range(donelen):
            tempidx = doneidx[i]
            idx[i] = idx2[tempidx, :numNN].tolist()
            dist[i] = dist2[tempidx, :numNN].tolist()
        idx = np.array(idx)
        dist = np.array(dist)

    if notdoneidx:
        print('Sorry for the code is wrong, please email to author and wait soon.')
        exit(1)
    return idx, dist


def similar(X, numN, kscale):
    """
    Construct a similarity matrix.
    :param X: Input data
    :param numN: A positive integer, specifying the number of nearest neighbors
                         used to construct the similarity graph
    :param kscale: Scale factor for the kernel
    """
    N = len(X)

    #  KNN graph generated using knn_search
    rows, weights = knn_search(X, numN + 1)

    # Construct KNN graph
    weights_temp = weights.flatten()
    weights_final = []
    for weight in weights_temp:
        weight = math.exp(-math.pow(weight/kscale, 2))
        weights_final.append(weight)
    cols = [1 for _ in range(len(weights_final))]
    num_temp = 0
    for idx in range(N):
        num = rows.shape[1]
        for colind in range(num):
            cols[colind + num_temp] = (idx + 1) * cols[colind + num_temp]
        num_temp += num
    rows = rows.flatten()
    S = np.zeros((N, N))
    for i in range(rows.shape[0]):
        S[rows[i], cols[i] - 1] = weights_final[i]

    # Make a 'complete' connection of points.
    SS = S.T
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i, j] < SS[i, j]:
                S[i, j] = SS[i, j]
    return S

