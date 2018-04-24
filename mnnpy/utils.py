"""."""
import numpy as np
from collections import Counter
from multiprocessing import Pool
from math import pi, acos, asin, sqrt
from numpy import matmul, unique, divide, mean, zeros, inner, exp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import norm, orth, svd
from scipy.linalg.interpolative import svd as rsvd
from numba import jit
from .irlb import lanczos


def l2_norm(in_matrix):
    return norm(a=in_matrix, axis=1)


def scale_rows(in_matrix, scale_vector):
    return divide(in_matrix, scale_vector[:, None])


def transform_input_data(datas, cos_norm_in, cos_norm_out, var_index, var_subset, n_jobs):
    if var_index is None:
        raise ValueError('Argument var_index not provideed.')
    if var_subset is not None:
        if set(var_subset) - set(var_index) != set():
            raise ValueError('Some items in var_subset are not in var_index.')
        do_subset = True
        if set(var_index) == set(var_subset):
            do_subset = False
    else:
        do_subset = False
    same_set = cos_norm_in == cos_norm_out and not do_subset
    if do_subset:
        var_sub_index = [list(var_index).index(var) for var in var_subset]
        in_batches = [data[:, var_sub_index] for data in datas]
    else:
        var_sub_index = None
        in_batches = datas
    with Pool(n_jobs) as p_n:
        in_scaling = p_n.map(l2_norm, in_batches)
    if cos_norm_in:
        with Pool(n_jobs) as p_n:
            in_batches = p_n.starmap(scale_rows, zip(in_batches, in_scaling))
    if cos_norm_out:
        if not cos_norm_in:
            with Pool(n_jobs) as p_n:
                out_batches = p_n.starmap(scale_rows, zip(datas, in_scaling))
        else:
            with Pool(n_jobs) as p_n:
                out_scaling = p_n.map(l2_norm, datas)
                out_batches = p_n.starmap(scale_rows, zip(datas, out_scaling))
    return in_batches, out_batches, var_sub_index, same_set


@jit(nogil=True)
def find_mutual_nn(data1, data2, k1, k2, n_jobs):
    # here nrows==nrows(data2)
    k_index_1 = cKDTree(data1).query(x=data2, k=k1, n_jobs=n_jobs)[1]
    # here nrows==nrows(data1)
    k_index_2 = cKDTree(data2).query(x=data1, k=k2, n_jobs=n_jobs)[1]
    mutual_1 = []
    mutual_2 = []
    for index_2 in range(data2.shape[0]):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
    return mutual_1, mutual_2


@jit(nogil=True)
def compute_correction(data1, data2, mnn1, mnn2, rawdata2, sigma):
    vect = data1[mnn1, :] - data2[mnn2, :]
    # data2 is not transposed
    if rawdata2 is None:
        return gaussian_kernel_smooth(vect, mnn2, data2, sigma)
    return gaussian_kernel_smooth(vect, mnn2, rawdata2, sigma)


@jit(nogil=True)
def gaussian_kernel_smooth(vect, mnn2, data2, sigma):
    mnn_index = unique(mnn2)
    mnn_count = np.array(list(Counter(mnn2).values()))
    vect_reduced = zeros((data2.shape[0], vect.shape[1]))
    for index, ve in zip(mnn2, vect):
        vect_reduced[index] += ve
    vect_avg = divide(vect_reduced[mnn_index], mnn_count[:, None])
    # exp_distance is n_cell * n_mnn
    exp_distance = exp(-cdist(data2, data2[mnn_index, :], 'sqeuclidean') / sigma)
    density = np.sum(exp_distance[mnn_index, :], axis=0)
    mult = divide(exp_distance, density)
    total_prob = np.sum(mult, axis=1, keepdims=True)
    output = matmul(mult, vect_avg)
    return divide(output, total_prob)


def svd_internal(mat, nu, svd_mode, **kwargs):
    if svd_mode == 'svd':
        svd_out = rsvd(mat, eps_or_k=nu, rand=False)
        return svd_out[0], svd_out[1], svd_out[2]
    if svd_mode == 'rsvd':
        svd_out = rsvd(mat, eps_or_k=nu)
        return svd_out[0], svd_out[1], svd_out[2]
    if svd_mode == 'irlb':
        svd_out = lanczos(mat, nu, **kwargs)
        return svd_out[0], svd_out[1], svd_out[2]
    raise ValueError('The svd_mode must be one of \'rsvd\', \'svd\', \'irlb\'.')


@jit(nogil=True)
def find_shared_subspace(mat1, mat2, sin_thres=0.05, cos_thres=1 / sqrt(2), mat2_vec=False,
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
            sintheta = norm(a=mat1 - matmul(mat2, cross_prod.T), ord=2)
        else:
            sintheta = norm(a=mat2.T - matmul(mat1, cross_prod), ord=2)
        theta = asin(min(1, sintheta))
    return 180 * theta / pi, shared


@jit(nogil=True)
def get_bio_span(exprs, ndim, svd_mode, var_subset=None, **kwargs):
    centred = exprs - mean(exprs, axis=0)
    if var_subset is not None:
        subsetter = [True] * centred.shape[1]
        keeper = [False] * centred.shape[1]
        for i in var_subset:
            subsetter[i] = False
            keeper[i] = True
        leftovers = centred[:, subsetter].T
        centred = centred[:, keeper]
    ndim = min(ndim, *centred.shape)
    singular = svd_internal(centred.T, ndim, svd_mode, **kwargs)
    if var_subset is None:
        return singular[0]
    output = zeros((exprs.shape[1], ndim))
    output[keeper,] = singular[0]
    output[subsetter,] = divide(matmul(leftovers, singular[2]), singular[1][range(ndim)])
    return output


@jit(nogil=True)
def subtract_bio(*spans, correction, var_subset=None):
    for span in spans:
        if var_subset is None:
            bio_mag = matmul(correction, span)
        else:
            bio_mag = matmul(correction[:, var_subset], span[:, var_subset])
        bio_comp = matmul(bio_mag, span.T)
        correction -= bio_comp
    return correction


def adjust_shift_variance(data1, data2, correction, sigma, n_jobs, var_subset=None):
    if var_subset is not None:
        vect = correction[:, var_subset]
        data1 = data1[:, var_subset]
        data2 = data2[:, var_subset]
    else:
        vect = correction
    with Pool(n_jobs) as p_n:
        scaling = p_n.starmap(adjust_v_worker(data1, data2, sigma), zip(data2, vect))
    scaling = max(*scaling, 1)
    return correction * scaling


@jit(nogil=True)
def adjust_s_variance(data1, data2, curcell, curvect, sigma):
    distance1 = zeros((data1.shape[0], 2))
    l2_norm = norm(curvect)
    grad = divide(curvect, l2_norm)
    curproj = inner(grad, curcell)
    prob2 = 0
    totalprob2 = 0
    for samecell in data2:
        sameproj = inner(grad, samecell)
        samedist = sq_dist_to_line(curcell, grad, samecell)
        sameprob = exp(-samedist / sigma)
        if sameproj <= curproj:
            prob2 += sameprob
        totalprob2 += sameprob
    prob2 /= totalprob2
    totalprob1 = 0
    for other in range(data1.shape[0]):
        othercell = data1[other]
        distance1[other, 0] = inner(grad, othercell)
        otherdist = sq_dist_to_line(curcell, grad, othercell)
        weight = exp(-otherdist / sigma)
        distance1[other, 1] = weight
        totalprob1 += weight
    distance1 = distance1[distance1[:, 0].argsort()]
    target = prob2 * totalprob1
    cumulative = 0
    ref_quan = distance1[-1, 0]
    for i in distance1:
        cumulative += i[1]
        if cumulative > target:
            ref_quan = i[0]
            break
    return (ref_quan - curproj) / l2_norm


@jit(nogil=True)
def sq_dist_to_line(ref, grad, point):
    working = ref - point
    scale = inner(working, grad)
    working = working - grad * scale
    return inner(working, working)


class adjust_v_worker(object):
    def __init__(self, data1, data2, sigma):
        self.d1 = data1
        self.d2 = data2
        self.s2 = sigma

    def __call__(self, curcell, curvect):
        return adjust_s_variance(self.d1, self.d2, curcell, curvect, self.s2)
