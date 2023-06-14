# -*- coding = utf-8 -*-
# The gap_evaluation functions for computing the gap criterion values
# which can be used for estimating the number of clusters.
# 2023.03.07,edit by Lin

import math
import numpy as np
from .ck_dist import calkagan
from .spectral_cluster import spectral_cluster


def get_svd(X):
    """
    Compute the svd of X, for PCA reference distribution.
    """
    mean_x = np.mean(X, 0)
    x0 = X - mean_x
    u, s, v = np.linalg.svd(x0)
    v_transform = v
    v = v.T
    x2 = np.dot(x0, v)
    min_x2 = np.min(x2, axis=0)
    max_x2 = np.max(x2, axis=0)
    return min_x2, max_x2, v_transform


def get_uniform_ref(min_x, max_x, n_rows, n_cols):
    """
    Generate reference data, which will be uniform between on the interval from minX and maxX.
    """
    ran = np.random.rand(n_rows, n_cols)

    repmat1 = np.tile((max_x - min_x), (n_rows, 1))
    repmat2 = np.tile(min_x, (n_rows, 1))
    xref = repmat2 + repmat1 * ran

    return xref


def p_dist(X):
    """
    Pairwise distance between observations.
    """
    n = X.shape[0]

    if n < 2:
        Y = np.zeros(1, dtype=X.dtype)
        return Y
    Y = np.zeros(int(n * (n - 1) / 2), dtype=X.dtype)
    k = 0
    for i in range(n - 1):
        for j in range(k, (k + n - i - 1)):
            Y[j] = calkagan(X[i].tolist(), X[i + j - k + 1].tolist())
        k += n - i - 1

    return Y


def get_log_w(X, idx):
    """
    Get log(W), which computed by the sum of pairwise distances divided the number of points.
    """
    p = X.shape[1]
    clusts = np.unique(idx)
    num = np.size(clusts)
    sum_D = np.zeros(num, dtype=X.dtype)

    for i in range(num):
        mem = X[idx == i]
        ni = np.size(mem, 0)
        p_dis = p_dist(mem)
        sum_D[i] = np.sum(p_dis) / ni

    log_w = math.log(np.sum(sum_D))

    return log_w


def get_ref_log_w(X, n_inspect=20, b_num=100, numN=150, kscale=40):
    """
    Get log(W) for reference data.
    """
    refLogW = np.zeros([b_num, n_inspect], dtype=X.dtype)
    n_rows = X.shape[0]
    n_cols = X.shape[1]

    min_x, max_x, v_transform = get_svd(X)
    print('Start for bootstrap loop...')
    for i in range(b_num):
        Xref = get_uniform_ref(min_x, max_x, n_rows, n_cols)
        Xref = np.dot(Xref, v_transform)
        for j in range(n_inspect):
            nc = j + 1
            try:
                idx = spectral_cluster(Xref, nc, numN=numN, kscale=kscale)
            except ValueError:
                refLogW[i][j] = np.nan
                continue
            refLogW[i][j] = get_log_w(Xref, idx)
        # print('The number of bootstrap is {}.'.format(str(i + 1)))
        print('%{} complete.'.format(str(int((i + 1) * 100 / b_num))))

    return refLogW


def get_ref_stats(refLogW):
    """
    Get the Expected value, STD, SE and the variance of log_w after ref_log_w computed.
    """
    idx_nan = []
    i_nan = np.isnan(refLogW[:, 0])
    for i in range(len(i_nan)):
        if i_nan[i]:
            idx_nan.append(i)
    refLogW = np.delete(refLogW, idx_nan, axis=0)
    num = refLogW.shape[0]

    exp_log_w = np.mean(refLogW, axis=0)
    std_log_w = np.sqrt(np.var(refLogW, axis=0))
    se = std_log_w * math.sqrt(1 / num + 1)

    return exp_log_w, std_log_w, se


def get_gap_value(X, nc, exp_log_w, numN=150, kscale=40):
    """
    Get the gap value and index by exp_log_w and log_w.
    """
    idx = spectral_cluster(X, nc, numN=numN, kscale=kscale)
    log_w = get_log_w(X, idx)
    criterion_values = exp_log_w - log_w
    return criterion_values, idx


def find_optimal(X, n_inspect=20, b_num=100, numN=150, kscale=40):
    """
    Compute the gap value (criterion), index, and the standard error for each number of clusters,
        then find the optimal number of clusters suggested (k) and the optimal clustering solution.
    :param X: Input data
    :param n_inspect: Using k that in range(n_inspect) for clustering
    :param b_num: The number of reference data sets used for computing gap values
    """
    optimal_k = -1
    optimal_y = np.full(X.shape[0], -1, dtype=np.int32)
    criterion_values = np.full(n_inspect, -1, dtype=X.dtype)
    idx = np.full([n_inspect, X.shape[0]], -1, dtype=np.int32)
    maxGap = float('-inf')
    th = float('-inf')

    ref_log_w = get_ref_log_w(X, n_inspect=n_inspect, b_num=b_num, numN=numN, kscale=kscale)
    exp_log_w, std_log_w, se = get_ref_stats(ref_log_w)
    print('Start to compute gap values...')

    for i in range(n_inspect - 1, -1, -1):
        nc = i + 1
        criterion_values[i], idx[i] = get_gap_value(X, nc, exp_log_w[i], numN=numN, kscale=kscale)

        if criterion_values[i] >= maxGap:
            maxGap = criterion_values[i]
            th = maxGap - se[i]
        if criterion_values[i] >= th:
            optimal_k = nc
            optimal_y = idx[i]
        print('%{} complete.'.format(str(int((n_inspect - i) * 100 / n_inspect))))

    return optimal_k, optimal_y, criterion_values, idx, se
