# -*- coding = utf-8 -*-
# Functions for spectral cluster
# 2022.09.01,edit by Lin

import math
import numpy as np
from scipy.sparse.linalg import eigs
from .similarity import similar
from .kmeans import kmeans


def start_vector(S, k):
    """
    validation and start vector for eigs.
    :param S: Similarity matrix
    :param k: Cluster number
    :return: Start vector
    """
    if k > S.shape[0]:
        raise ValueError('Error: Too many clusters!')
    sv = np.random.randn(S.shape[0], 1)
    return sv


def shi_malik(S, k, sv):
    """
    Shi-Malik algorithm for computing Laplacian, eigenvectors and eigenvalues
    :param S: Similarity matrix
    :param k: Cluster number
    :param sv: Start vector
    :return: D-eigenvalues; V-eigenvectors; L-Laplacian matrix
    """
    L, Deg = get_degree_and_laplacian(S)
    Deg = invert_degree_matrix(Deg)
    Deg = Deg.reshape(len(Deg), 1)
    L = Deg * L
    L = set_subnormal_nums_to_zero(L)
    D, V = eigs(L, k, sigma=1e-10, v0=sv, tol=1e-14, maxiter=300, which='LM')
    return D, V, L


def get_degree_and_laplacian(L):
    """
    Get the degree and unnormalized Laplacian matrices.
    """
    j = 0
    for i in range(L.shape[0]):
        L[i, j] = 0
        j += 1
    D = np.sum(L, axis=1)
    L = -L
    L = L + np.diag(D)
    return L, D


def invert_degree_matrix(d):
    """
    Degree matrix inversion (pseudo-inverse).
    """
    N = np.size(d)
    dd = 1. / d
    for i in range(N):
        if d[i] <= np.spacing(np.max(d)):
            dd[i] = 0
    return dd


def set_subnormal_nums_to_zero(L):
    """
    Remove subnormal numbers from Laplacian matrix.
    """
    N = np.size(L)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if N * np.spacing(np.max(L)) >= math.fabs(L[i, j]) > 0:
                L[i, j] = 0
    return L


def spectral_cluster(data, k, numN=150, kscale=40, s_log=False):
    """
    Main function for spectral cluster.
    :param data: Input data
    :param k: Cluster number
    :param numN: The number of nearest neighbors for KNN
    :param kscale: Scale factor for the kernel
    :param s_log: bool; The spectral cluster print show or not
    :return: The index for cluster results
    """
    new_list = data.tolist()
    if s_log:
        print('Begin to construct a similarity matrix...')
    S = similar(new_list, numN=numN, kscale=kscale)
    if s_log:
        print('Begin to spectral cluster...')
    sv = start_vector(S, k)
    D, V, L = shi_malik(S, k, sv)
    X = V.real

    labels = kmeans(X, k, n_init=20, max_iter=500)

    return labels
