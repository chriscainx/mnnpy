"""."""
import numpy as np
from multiprocessing import Pool
from collections import Counter
from math import pi, acos, asin, sqrt
from numpy import matmul, unique, divide, mean, zeros, inner, exp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import norm, orth, svd
from scipy.linalg.interpolative import svd as rsvd
from sppy.linalg import rsvd as sppy_rsvd
from .irlbpy import lanczos


def cosine_norm(in_matrix):
    l2_norm = norm(x=in_matrix, axis=1)
    #l2norm = np.fmax(l2norm, 0.00000001)
    out_matrix = divide(in_matrix, l2_norm[:, None])
    return out_matrix

def l2_norm(in_matrix):
    return norm(x=in_matrix, axis=1)

def scale_rows(in_matrix, scale_vector):
    return divide(in_matrix, scale_vector[:, None])

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
    vect_reduced = zeros(data2.shape)
    for index, ve in zip(mnn2, vect):
        vect_reduced[index] += ve
    vect_avg = divide(vect_reduced[mnn_index],
                      mnn_count[:, None])
    # exp_distance is n_cell * n_mnn
    exp_distance = exp(-cdist(data2, data2[mnn_index, :], 'seuclidian')/sigma)
    density = np.sum(exp_distance[mnn_index, :], axis=0)
    mult = divide(exp_distance, density)
    total_prob = np.sum(mult, axis=1, keepdims=True)
    output = matmul(mult, vect_avg)
    return divide(output, total_prob)

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
            sintheta = norm(mat1 - matmul(mat2, cross_prod.T), 2)
        else:
            sintheta = norm(mat2.T - matmul(mat1, cross_prod), 2)
        theta = asin(min(1, sintheta))
    return 180*theta/pi, shared

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
    output[keeper, ] = singular[0]
    output[subsetter, ] = divide(matmul(leftovers, singular[2]), singular[1][range(ndim)])
    return output

def subtract_bio(*spans, correction, var_subset=None):
    for span in spans:
        if var_subset is None:
            bio_mag = matmul(correction, span)
        else:
            bio_mag = matmul(correction[:, var_subset], span[:, var_subset])
        bio_comp = matmul(bio_mag, span.T)
        correction -= bio_comp
    return correction

def adjust_shift_variance(data1, data2, correction, sigma, var_subset=None):
    cell_vect = correction
    if var_subset is not None:
        cell_vect = cell_vect[:, var_subset]
        data1 = data1[:, var_subset]
        data2 = data2[:, var_subset]
    scaling = adjust_s_variance(data1, data2, cell_vect, sigma)
    scaling = max(*scaling, 1)
    return correction * scaling

def adjust_s_variance(data1, data2, vect, sigma):
    distance1 = zeros((data1.shape[0], 2))
    output = zeros((data2.shape[0]))
    for cell in range(data2.shape[0]):
        curcell = data2[cell]
        curvect = vect[cell]
        l2_norm = norm(curvect)
        grad = divide(curvect, norm)
        curproj = inner(grad, curcell)
        #-----------------------------
        prob2 = totalprob2 = 0
        for same in range(data2.shape[0]):
            if same == cell:
                prob2 += 1
                totalprob2 += 1
            else:
                samecell = data2[same]
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
        #std::sort(distance1.begin(), distance1.end());
        # sorts distance[,0] in ascending order
        distance1 = distance1[distance1[:, 0].argsort()]
        target = prob2 * totalprob1
        cumulative = ref_quan = 0
        if data1.shape[0] > 0:
            ref_quan = distance1[-1, 0]
        for i in distance1:
            cumulative += i[1]
            if cumulative > target:
                ref_quan = i[0]
                break
        output[cell] = (ref_quan - curproj) / l2_norm
    return output

def sq_dist_to_line(ref, grad, point):
    working = ref - point
    scale = inner(working, grad)
    working = working - grad * scale
    dist = sum([x * x for x in working])
    return dist

# C++ implementations
#try:
#    from ._utils import cosine_norm, find_mutual_nn, gaussian_kernel_smooth
#    from ._utils import adjust_s_variance
#except ImportError:
#    pass
