cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free, qsort
from libc.math cimport exp, sqrt
import numpy as np

cdef extern from "_utils.h":
    int comp(const void* a, const void* b) nogil

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef float _adjust_s_variance(float [:, :] data1, float [:, :] data2, float [:] curcell, float [:] curvect, float sigma) nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n_c1  = data1.shape[0]
    cdef Py_ssize_t n_c2  = data2.shape[0]
    cdef Py_ssize_t n_v   = data2.shape[1]
    cdef float prob2      = 0
    cdef float totalprob2 = 0
    cdef float curproj    = 0
    cdef float l2_norm    = 0
    for j in range(n_v):
        l2_norm += curvect[j] * curvect[j]
    l2_norm = sqrt(l2_norm)
    cdef float *grad = <float *>malloc(n_v * sizeof(float))
    for j in range(n_v):
        grad[j] = curvect[j] / l2_norm
        curproj += grad[j] * curcell[j]
    cdef float **d1 = <float **>malloc(n_c1 * sizeof(float*))
    for i in range(n_c1):
        d1[i] = <float *>malloc(2 * sizeof(float))
    cdef float sameproj
    cdef float samedist
    cdef float dist_scale
    cdef float *dist_working = <float *>malloc(n_v * sizeof(float))
    cdef float sameprob
    for i in range(n_c2):
        sameproj   = 0
        samedist   = 0
        dist_scale = 0
        for j in range(n_v):
            sameproj += grad[j] * data2[i, j]
            dist_working[j] = curcell[j] - data2[i, j]
        for j in range(n_v):
            dist_scale += dist_working[j] * grad[j]
        for j in range(n_v):
            dist_working[j] -= grad[j] * dist_scale
            samedist += dist_working[j] * dist_working[j]
        sameprob = exp(-samedist / sigma)
        if sameproj <= curproj:
            prob2 += sameprob
        totalprob2 += sameprob
    prob2 /= totalprob2
    cdef float totalprob1 = 0
    cdef float otherdist
    cdef float weight
    for i in range(n_c1):
        otherdist  = 0
        dist_scale = 0
        d1[i][0] = 0
        for j in range(n_v):
            d1[i][0] += grad[j] * data1[i, j]
            dist_working[j] = curcell[j] - data1[i, j]
        for j in range(n_v):
            dist_scale += dist_working[j] * grad[j]
        for j in range(n_v):
            dist_working[j] -= grad[j] * dist_scale
            otherdist += dist_working[j] * dist_working[j]
        weight = exp(-otherdist / sigma)
        d1[i][1] = weight
        totalprob1 += weight
    qsort(d1, n_c1, sizeof(float*), &comp)
    cdef float target = prob2 * totalprob1
    cdef float cumulative = 0
    cdef float ref_quan = d1[n_c1 - 1][0]
    for i in range(n_c1):
        cumulative += d1[i][1]
        if cumulative > target:
            ref_quan = d1[i][0]
            break
    free(grad)
    free(dist_working)
    for i in range(n_c1):
        free(d1[i])
    free(d1)
    return (ref_quan - curproj) / l2_norm

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef _adjust_shift_variance(data1, data2, correction, sigma, n_jobs, var_subset=None):
    if var_subset is not None:
        vect = correction[:, var_subset]
        data1 = data1[:, var_subset]
        data2 = data2[:, var_subset]
    else:
        vect = correction
    cdef float [:, :] da1 = data1
    cdef float [:, :] da2 = data2
    cdef float [:, :] vec = vect
    scaling = np.zeros(data2.shape[0], dtype=np.float32)
    cdef float [:] s1 = scaling
    cdef float s2 = sigma
    cdef Py_ssize_t i
    cdef Py_ssize_t n_c2  = data2.shape[0]
    cdef int chunksize=int(n_c2/n_jobs) + 1
    for i in prange(n_c2, nogil=True, chunksize=chunksize, schedule='static'):
        s1[i] = _adjust_s_variance(da1, da2, da2[i], vec[i], s2)
    scaling = np.fmax(scaling, 1)
    return correction * scaling[:, None]
