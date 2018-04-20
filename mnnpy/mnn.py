from multiprocessing import Pool
from os import cpu_count
from anndata import AnnData
from .utils import cosine_norm, l2_norm, scale_rows, find_mutual_nn, compute_correction


def mnn_correct(*datas, var_index=None, var_subset=None, batch_key=None, index_unique='-',
                batch_categories=None, k=20, sigma=1, cos_norm_in=True, cos_norm_out=True,
                svd_dim=0, var_adj=True, compute_angle=False, mnn_order=None, pc_approx=True,
                do_concatenate=True, save_raw=False, n_jobs=None, **kwargs):
    if len(datas) < 2:
        return datas
    # Rows are cells, since by default NumPy is row-majored (C mode).
    # if Armadillo is to be integrated, will have to change to F mode
    n_batch = len(datas)
    if mnn_order is not None:
        if sorted(mnn_order) != list(range(n_batch)):
            raise ValueError('The argument mnn_order should contain values in 1:'+'n_batch'+'.')
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
                                compute_angle=compute_angle, mnn_order=mnn_order,
                                pc_approx=pc_approx, do_concatenate=do_concatenate, **kwargs)
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
    #------------------------------------------------------------
    # adatas are ndarrays, return a ndarray
    # by definition ndarrays don't have colnames
    n_cols = datas[0].shape[1]
    if len(var_index) != n_cols:
        raise ValueError('The number of vars is not equal to the length of var_index.')
    for i in range(1, n_batch):
        if datas[i].shape[1] != n_cols:
            raise ValueError('The input matrices have inconsistent number of columns.')
    #prep_out = prepare_input_data(*datas, var_index=var_index, var_subset=var_subset,
    #                              cos_norm_in=cos_norm_in, cos_norm_out=cos_norm_out)
    # in_batches = subset datas
    #------------------------------------------------------------
    if n_jobs is None:
        n_jobs = cpu_count()
    if n_jobs is None:
        n_jobs = 1
    if var_index is None:
        raise ValueError('Argument var_index not provideed.')
    if set(var_subset) - set(var_index) != set():
        raise ValueError('Some items in var_subset are not in var_index.')
    do_subset = True
    if set(var_index) == set(var_subset):
        do_subset = False
    same_set = cos_norm_in == cos_norm_out and not do_subset
    out_batches = datas
    if do_subset:
        in_batches = [data[:, [var_index.index(var) for var in var_subset]] for data in datas]
    else:
        in_batches = datas
    if cos_norm_in:
        with Pool(n_jobs) as p_n:
            in_batches = p_n.map(cosine_norm, in_batches)
    if cos_norm_out:
        if not cos_norm_in:
            with Pool(n_jobs) as p_n:
                norm_vectors = p_n.map(l2_norm, in_batches)
        with Pool(n_jobs) as p_n:
            out_batches = p_n.map(scale_rows, out_batches, norm_vectors)
    #------------------------------------------------------------
    if mnn_order is None:
        mnn_order = list(range(n_batch))
    ref = mnn_order[0]
    ref_batch_in = in_batches[ref]
    if not same_set:
        ref_batch_out = out_batches[ref]
    # prepare a container for results
    # add results by res_container.append()
    res_container = [out_batches[ref]]
    mnn_list = [0] * n_batch
    angle_list = [0] * n_batch
    original_batch = [ref] * ref_batch_in.shape[0]
    #------------------------------------------------------------
    # loop through batches
    for step in range(1, n_batch):
        target = mnn_order[step]
        new_batch_in = in_batches[target]
        if not same_set:
            new_batch_out = out_batches[target]
        mnn_ref, mnn_new = find_mutual_nn(data1=ref_batch_in, data2=new_batch_in, k1=k, k2=k,
                                          n_jobs=n_jobs)
        correction_in = compute_correction(ref_batch_in, new_batch_in, mnn_ref, mnn_new, sigma)

    merged_mat = mats = 1
    if do_concatenate:
        return merged_mat
    return mats

