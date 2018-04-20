from anndata import AnnData
from .utils import *

def mnn_correct(*datas, var_index=None, var_subset=None, batch_key=None, index_unique='-',
                batch_categories=None, k=20, sigma=1, cos_norm_in=True, cos_norm_out=True,
                svd_dim=0, var_adj=True, compute_angle=False, order=None, pc_approx=True,
                do_concatenate=True, save_raw=False, **kwargs):
    if len(datas) < 2:
        return datas
    do_subset = True
    if set(var_index) == set(var_subset):
        var_subset = None
        do_subset = False
    if isinstance(datas[0], AnnData):
        if var_index is not None:
            print('Inputs are AnnData objects, var_index ignored.')
        n_batch = len(datas)
        adata_vars = datas[0].var.index
        for i in range(1, n_batch):
            if datas[i].var.index != adata_vars:
                raise ValueError('The AnnData objects have inconsistent number of vars.')
        if set(adata_vars) == set(var_subset):
            var_subset = None
        # return a tuple of matrices if do_concatenate==False, else a combined matrix
        corrected = mnn_correct(*(adata.X for adata in datas), var_index=adata_vars,
                                var_subset=var_subset, k=k, sigma=sigma, cos_norm_in=cos_norm_in,
                                cos_norm_out=cos_norm_out, svd_dim=svd_dim, var_adj=var_adj,
                                compute_angle=compute_angle, order=order, pc_approx=pc_approx,
                                do_concatenate=do_concatenate, **kwargs)
        if do_concatenate:
            # concatenate and replace X with the combined matrix
            adata = AnnData.concatenate(*datas, batch_key=batch_key,
                                        batch_categories=batch_categories,
                                        index_unique=index_unique)
            if save_raw:
                adata.raw = adata.copy()
            if var_subset is not None:
                adata = adata[:, var_subset]
            adata.X = corrected
            return adata
        else:
            for adata, new_matrix in zip(datas, corrected):
                if save_raw:
                    adata.raw = adata.copy()
                if var_subset is not None:
                    adata = adata[:, var_subset]
                adata.X = new_matrix
            return datas
    # adatas are ndarrays, return a ndarray
    # by definition ndarrays don't have colnames
    n_batch = len(datas)
    n_cols = datas[0].shape[1]
    for i in range(1, n_batch):
        if datas[i].shape[1] != n_cols:
            raise ValueError('The input matrices have inconsistent number of columns.')
    #prep_out = prepare_input_data(*datas, var_index=var_index, var_subset=var_subset,
    #                              cos_norm_in=cos_norm_in, cos_norm_out=cos_norm_out)
    # in_batches = subset datas
    out_batches = datas
    if do_subset:
        in_batches = [data[:, [var_index.index(var) for var in var_subset]] for data in datas]
    else:
        in_batches = datas
    merged_mat = mats = 1
    if do_concatenate:
        return merged_mat
    return mats

