import math
import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree
from scipy.linalg import orth
from scipy.linalg.interpolative import svd as rsvd
from scipy.sparse import issparse
from numba import jit, float32, int32, int8
from . import settings
from .irlb import lanczos



@jit(float32[:](float32[:, :]), nogil=True)
def l2_norm(in_matrix):
    return np.linalg.norm(x=in_matrix, axis=1)


@jit(float32[:, :](float32[:, :], float32[:, :]), nogil=True)
def scale_rows(in_matrix, scale_vector):
    return np.divide(in_matrix, scale_vector)


@jit(float32[:, :](float32[:, :], float32[:, :]))
def kdist(m, n):
    dist = np.zeros((m.shape[0], n.shape[0]), dtype=np.float32)
    for i in range(m.shape[0]):
        for j in range(n.shape[0]):
            dist[i, j] = np.dot(m[i], n[j])
    return dist


def transform_input_data(datas, cos_norm_in, cos_norm_out, var_index, var_subset, n_jobs):
    datas = [data.toarray().astype(np.float32) if issparse(data) else data.astype(np.float32) for data in datas]
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
    if settings.normalization == 'parallel':
        with Pool(n_jobs) as p_n:
            in_scaling = p_n.map(l2_norm, in_batches)
    else:
        in_scaling = [l2_norm(b) for b in in_batches]
    in_scaling = [scaling[:, None] for scaling in in_scaling]
    if cos_norm_in:
        if settings.normalization == 'parallel':
            with Pool(n_jobs) as p_n:
                in_batches = p_n.starmap(scale_rows, zip(in_batches, in_scaling))
        else:
            in_batches = [scale_rows(a,b) for (a,b) in zip(in_batches, in_scaling)]
    if cos_norm_out:
        if not cos_norm_in:
            if settings.normalization == 'parallel':
                with Pool(n_jobs) as p_n:
                    out_batches = p_n.starmap(scale_rows, zip(datas, in_scaling))
            else:
                out_batches = [scale_rows(a,b) for (a,b) in zip(datas, in_scaling)]
        else:
            if settings.normalization == 'parallel':
                with Pool(n_jobs) as p_n:
                    out_scaling = p_n.map(l2_norm, datas)
            else:
                out_scaling = [l2_norm(d) for d in datas]
            out_scaling = [scaling[:, None] for scaling in out_scaling]
            if settings.normalization == 'parallel':
                with Pool(n_jobs) as p_n:
                    out_batches = p_n.starmap(scale_rows, zip(datas, out_scaling))
            else:
                out_batches = [scale_rows(a,b) for (a,b) in zip(datas, out_scaling)]
    else: 
        out_batches = datas
    return in_batches, out_batches, var_sub_index, same_set


@jit((float32[:, :], float32[:, :], int8, int8, int8))
def find_mutual_nn(data1, data2, k1, k2, n_jobs):
    k_index_1 = cKDTree(data1).query(x=data2, k=k1, n_jobs=n_jobs)[1]
    k_index_2 = cKDTree(data2).query(x=data1, k=k2, n_jobs=n_jobs)[1]
    mutual_1 = []
    mutual_2 = []
    for index_2 in range(data2.shape[0]):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
    return mutual_1, mutual_2


@jit(float32[:, :](float32[:, :], float32[:, :], int32[:], int32[:], float32[:, :], float32))
def compute_correction(data1, data2, mnn1, mnn2, data2_or_raw2, sigma):
    vect = data1[mnn1] - data2[mnn2]
    mnn_index, mnn_count = np.unique(mnn2, return_counts=True)
    vect_reduced = np.zeros((data2.shape[0], vect.shape[1]), dtype=np.float32)
    for index, ve in zip(mnn2, vect):
        vect_reduced[index] += ve
    vect_avg = np.divide(vect_reduced[mnn_index], mnn_count.astype(np.float32)[:, None])
    exp_distance = np.exp(-kdist(data2_or_raw2, data2_or_raw2[mnn_index]) / sigma)
    density = np.sum(exp_distance[mnn_index], axis=0)
    mult = np.divide(exp_distance, density)
    total_prob = np.sum(mult, axis=1, keepdims=True)
    output = np.dot(mult, vect_avg)
    return np.divide(output, total_prob)


def svd_internal(mat, nu, svd_mode, **kwargs):
    mat = mat.astype(np.float64)
    if svd_mode == 'svd':
        svd_out = rsvd(mat, eps_or_k=nu, rand=False)
    elif svd_mode == 'rsvd':
        svd_out = rsvd(mat, eps_or_k=nu)
    elif svd_mode == 'irlb':
        svd_out = lanczos(mat, nu, **kwargs)
    else:
        raise ValueError('The svd_mode must be one of \'rsvd\', \'svd\', \'irlb\'.')
    return svd_out[0].astype(np.float32), svd_out[1].astype(np.float32), svd_out[2].astype(np.float32)

def find_shared_subspace(mat1, mat2, sin_thres=0.05, cos_thres=1 / math.sqrt(2), mat2_vec=False,
                         assume_orthonomal=False, get_angle=True):
    if mat2_vec:
        mat2 = mat2[:, None]
    if not assume_orthonomal:
        mat1 = orth(mat1)
        mat2 = orth(mat2)
    cross_prod = np.dot(mat1.T, mat2)
    singular = np.linalg.svd(cross_prod)
    shared = sum(singular[1] > sin_thres)
    if not get_angle:
        return None, shared
    costheta = min(singular[1])
    if costheta < cos_thres:
        theta = math.acos(min(1, costheta))
    else:
        if mat1.shape[1] < mat2.shape[1]:
            sintheta = np.linalg.norm(x=mat1 - np.dot(mat2, cross_prod.T), ord=2)
        else:
            sintheta = np.linalg.norm(x=mat2.T - np.dot(mat1, cross_prod), ord=2)
        theta = math.asin(min(1, sintheta))
    return 180 * theta / math.pi, shared


def get_bio_span(exprs, ndim, svd_mode, var_subset=None, **kwargs):
    centred = exprs - np.mean(exprs, axis=0)
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
    output = np.zeros((exprs.shape[1], ndim), dtype=np.float32)
    output[keeper,] = singular[0]
    output[subsetter,] = np.divide(np.dot(leftovers, singular[2]), singular[1][range(ndim)])
    return output


def subtract_bio(*spans, correction, var_subset=None):
    for span in spans:
        if var_subset is None:
            bio_mag = np.dot(correction, span)
        else:
            bio_mag = np.dot(correction[:, var_subset], span[var_subset, :])
        bio_comp = np.dot(bio_mag, span.T)
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
        scaling = p_n.starmap(adjust_v_worker(data1, data2, sigma), zip(data2, vect), 
                              chunksize=int(data2.shape[0]/n_jobs) + 1)
    scaling = np.fmax(scaling, 1).astype(np.float32)
    return correction * scaling[:, None]


@jit(float32(float32[:, :], float32[:, :], float32[:], float32[:], float32), nogil=True)
def adjust_s_variance(data1, data2, curcell, curvect, sigma):
    distance1 = np.zeros((data1.shape[0], 2), dtype=np.float32)
    l2_norm = np.linalg.norm(curvect)
    grad = np.divide(curvect, l2_norm)
    curproj = np.dot(grad, curcell)
    prob2 = 0.
    totalprob2 = 0.
    for samecell in data2:
        sameproj = np.dot(grad, samecell)
        samedist = sq_dist_to_line(curcell, grad, samecell)
        sameprob = np.exp(-samedist / sigma)
        if sameproj <= curproj:
            prob2 += sameprob
        totalprob2 += sameprob
    prob2 /= totalprob2
    totalprob1 = 0.
    for other in range(data1.shape[0]):
        othercell = data1[other]
        distance1[other, 0] = np.dot(grad, othercell)
        otherdist = sq_dist_to_line(curcell, grad, othercell)
        weight = np.exp(-otherdist / sigma)
        distance1[other, 1] = weight
        totalprob1 += weight
    distance1 = distance1[distance1[:, 0].argsort()]
    target = prob2 * totalprob1
    cumulative = 0.
    ref_quan = distance1[-1, 0]
    for i in distance1:
        cumulative += i[1]
        if cumulative > target:
            ref_quan = i[0]
            break
    return (ref_quan - curproj) / l2_norm


@jit(float32(float32[:], float32[:], float32[:]), nopython=True)
def sq_dist_to_line(ref, grad, point):
    working = ref - point
    scale = np.dot(working, grad)
    working = working - grad * scale
    return np.dot(working, working)


class adjust_v_worker(object):
    def __init__(self, data1, data2, sigma):
        self.d1 = data1
        self.d2 = data2
        self.s2 = sigma

    def __call__(self, curcell, curvect):
        return adjust_s_variance(self.d1, self.d2, curcell, curvect, self.s2)


def get_so_paths(dir_name):
    dir_name = os.path.join(os.path.dirname(__file__), dir_name)
    list_dir = os.listdir(dir_name) if os.path.isdir(dir_name) else []
    return [os.path.join(dir_name, so_name) for so_name in list_dir if so_name.split('.')[-1] in ['so', 'pyd']]



try:
    from ._utils import _adjust_shift_variance as adjust_shift_variance
    #print('Cython module loaded!')
except ImportError:
    print('Cython module _utils not initialized. Fallback to python.')
    pass
