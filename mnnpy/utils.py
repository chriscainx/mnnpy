"""."""
import numpy as np
from multiprocessing import Pool
from collections import Counter
from math import pi, acos, asin, sqrt
from numpy import matmul, unique
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import norm, orth, svd
from scipy.linalg.interpolative import svd as rsvd
from sppy.linalg import rsvd as sppy_rsvd
from irlbpy import lanczos


def cosine_norm(in_matrix):
    l2_norm = norm(x=in_matrix, axis=1)
    #l2norm = np.fmax(l2norm, 0.00000001)
    out_matrix = np.divide(in_matrix, l2_norm[:, None])
    return out_matrix

def l2_norm(in_matrix):
    return norm(x=in_matrix, axis=1)

def scale_rows(in_matrix, scale_vector):
    return np.divide(in_matrix, scale_vector[:, None])

def find_mutual_nn(data1, data2, k1, k2, n_jobs):
    # here nrows==nrows(data2)
    k_index_1 = cKDTree(data1).query(x=data2, k=k1, n_jobs=n_jobs)[1]
    # here nrows==nrows(data1)
    k_index_2 = cKDTree(data2).query(x=data1, k=k2, n_jobs=n_jobs)[1]
    mutual_1 = mutual_2 = []
    for index_2 in range(data2.shape[0]):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
    return mutual_1, mutual_2

def compute_correction(data1, data2, mnn1, mnn2, rawdata2, sigma):
    vect = data1[mnn1, :] - data2[mnn2, :]
    # data2 is not transposed
    if rawdata2 is None:
        return gaussian_kernel_smooth(vect, mnn2, data2, sigma)
    return gaussian_kernel_smooth(vect, mnn2, rawdata2, sigma)

def gaussian_kernel_smooth(vect, mnn2, data2, sigma):
    mnn_index = unique(mnn2)
    mnn_count = np.array(list(Counter(mnn2).values()))
    vect_reduced = np.zeros(data2.shape)
    for index, v in zip(mnn2, vect):
        vect_reduced[index] += v
    vect_avg = np.divide(vect_reduced[mnn_index], 
                         mnn_count[:, None])
    # exp_distance is n_cell * n_mnn
    exp_distance = np.exp(-cdist(data2, data2[mnn_index, :], 'seuclidian')/sigma)
    density = np.sum(exp_distance[mnn_index, :], axis=0)
    mult = np.divide(exp_distance, density)
    total_prob = np.sum(mult, axis=1, keepdims=True)
    output = matmul(mult, vect_avg)
    return np.divide(output, total_prob)

def svd_internal(mat, nu, svd_mode, **kwargs):
    if svd_mode == 'sppy_rsvd':
        svd_out = sppy_rsvd(mat, k=nu, **kwargs)
        return svd_out[0], svd_out[1], svd_out[2]
    if svd_mode == 'rsvd':
        svd_out = rsvd(mat, eps_or_k=nu, **kwargs)
        return svd_out[0], svd_out[1], svd_out[2]
    if svd_mode == 'irlb':
        svd_out = lanczos(mat, nu, **kwargs)
        return svd_out[0], svd_out[1], svd_out[2]
    raise ValueError('The svd_mode must be one of \'rsvd\', \'svd\', \'irlb\'.')

def find_shared_subspace(mat1, mat2, sin_thres=0.05, cos_thres=1/sqrt(2), mat2_vec=False,
                         assume_orthonomal=False, get_angle=True):
    if mat2_vec:
        mat2 = mat2[:, None]
    if not assume_orthonomal:
        mat1 = orth(mat1)
        mat2 = orth(mat2)
    cross_prod = matmul(mat1.T, mat2)
    singular = svd(cross_prod)
    shared = sum(singular[1] > sin_thres)
    if not get_angle:
        return None, shared
    costheta = min(singular[1])
    if costheta < cos_thres:
        theta = acos(min(1, costheta))
    else:
        if mat1.shape[1] < mat2.shape[1]:
            sintheta = rsvd(mat1.T - matmul(cross_prod, mat2.T), 1)[1][0]
        else:
            sintheta = rsvd(mat2.T - matmul(mat1, cross_prod).T, 1)[1][0]
        theta = asin(min(1, sintheta))
    return 180*theta/pi, shared

def get_bio_span(exprs, ndim, subset_row=None, svd_mode='rsvd', **kwargs):

def subtract_bio():

def adjust_shift_variance():



# C++ implementations
try:
    from ._utils import cosine_norm, find_mutual_nn, gaussian_kernel_smooth
except ImportError:
    pass
