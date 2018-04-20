"""."""
import numpy as np
#from multiprocessing import Pool
from scipy.spatial import cKDTree

def cosine_norm(in_matrix):
    l2_norm = np.linalg.norm(x=in_matrix, axis=1)
    #l2norm = np.fmax(l2norm, 0.00000001)
    out_matrix = np.divide(in_matrix, l2_norm[:, None])
    return out_matrix

def l2_norm(in_matrix):
    return np.linalg.norm(x=in_matrix, axis=1)

def scale_rows(in_matrix, scale_vector):
    return np.divide(in_matrix, scale_vector[:, None])

def find_mutual_nn(data1, data2, k1, k2, n_jobs):
    k_index_1 = cKDTree(data1).query(x=data2, k=k1, n_jobs=n_jobs)[1]
    k_index_2 = cKDTree(data2).query(x=data1, k=k2, n_jobs=n_jobs)[1]
    mutual_1 = mutual_2 = []
    for index_2 in data2.shape[0]:
        for index_1 in data2[index_2]:
            if index_2 in data1[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
    return mutual_1, mutual_2

def compute_correction()

# C++ implementations
try:
    from ._utils import cosine_norm, find_mutual_nns
except ImportError:
    pass
