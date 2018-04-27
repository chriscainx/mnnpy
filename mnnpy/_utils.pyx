cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport exp, sqrt
import numpy as np

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

cdef float adjust_s_variance(float [:, ::1] data1, float [:, ::1] data2, float [:] curcell, float [:] curvect, float sigma):
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
    distance1 = np.zeros((n_c1, 2), dtype=np.float32)
    cdef float [:, :] d1 = distance1
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
        for j in range(n_v):
            d1[i, 0] += grad[j] * data1[i, j]
            dist_working[j] = curcell[j] - data1[i, j]
        for j in range(n_v):
            dist_scale += dist_working[j] * grad[j]
        for j in range(n_v):
            dist_working[j] -= grad[j] * dist_scale
            otherdist += dist_working[j] * dist_working[j]
        weight = exp(-otherdist / sigma)
        d1[i, 1] = weight
        totalprob1 += weight
    distance1 = distance1[distance1[:, 0].argsort()]
    cdef float target = prob2 * totalprob1
    cdef float cumulative = 0
    cdef float ref_quan = d1[n_c1 - 1, 0]
    for i in range(n_c1):
        cumulative += d1[i, 1]
        if cumulative > target:
            ref_quan = d1[i, 0]
            break
    free(grad)
    free(dist_working)
    return (ref_quan - curproj) / l2_norm