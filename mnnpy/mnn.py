from os import cpu_count
from multiprocessing import Pool
from functools import partial
import numpy as np
from anndata import AnnData
from pandas import DataFrame
from .utils import transform_input_data, find_mutual_nn, compute_correction
from .utils import svd_internal, find_shared_subspace, get_bio_span, subtract_bio
from .utils import adjust_shift_variance


def mnn_correct(*datas, var_index=None, var_subset=None, batch_key='batch', index_unique='-',
                batch_categories=None, k=20, sigma=1., cos_norm_in=True, cos_norm_out=True,
                svd_dim=None, var_adj=True, compute_angle=False, mnn_order=None, svd_mode='rsvd',
                do_concatenate=True, save_raw=False, n_jobs=None, **kwargs):
    """
    Apply MNN correct to input data matrices or AnnData objects. Depending on do_concatenate,
    returns matrices or AnnData objects in the original order containing corrected expression
    values, or concatenated matrices or AnnData object.

    :param datas: `numpy.ndarray` or class:`anndata.AnnData`
        Expression matrices or AnnData objects. Matrices should be shaped like n_obs * n_vars
        (n_cell * n_gene) and have consistent number of columns. AnnData objects should have same
        number of vars.

    :param var_index: `list` or `None`, optional (default: None)
        The index (list of str) of vars (genes). Necessary when using only a subset of vars to
        perform MNN correction, and should be supplied with var_subset. When datas are AnnData
        objects, var_index is ignored.

    :param var_subset: `list` or `None`, optional (default: None)
        The subset of vars (list of str) to be used when performing MNN correction. Typically, a
        list of highly variable genes (HVGs). When set to None, uses all vars.

    :param batch_key: `str`, optional (default: 'batch')
        The batch_key for AnnData.concatenate. Only valid when do_concatenate and supplying AnnData
        objects.

    :param index_unique: `str`, optional (default: '-')
        The index_unique for AnnData.concatenate. Only valid when do_concatenate and supplying
        AnnData objects.

    :param batch_categories: `list` or `None`, optional (default: None)
        The batch_categories for AnnData.concatenate. Only valid when do_concatenate and supplying
        AnnData objects.

    :param k: `int`, optional (default: 20)
        Number of mutual nearest neighbors.

    :param sigma: `float`, optional (default: 1)
        The bandwidth of the Gaussian smoothing kernel used to compute the correction vectors.

    :param cos_norm_in: `bool`, optional (default: True)
        Whether cosine normalization should be performed on the input data prior to calculating
        distances between cells.

    :param cos_norm_out: `bool`, optional (default: True)
        Whether cosine normalization should be performed prior to computing corrected expression
        values.

    :param svd_dim: `int` or `None`, optional (default: None)
        The number of dimensions to use for summarizing biological substructure within each batch.
        If set to None, biological components will not be removed from the correction vectors.

    :param var_adj: `bool`, optional (default: True)
        Whether to adjust variance of the correction vectors. Note this step takes most computing
        time.

    :param compute_angle: `bool`, optional (default: False)
        Whether to compute the angle between each cell’s correction vector and the biological
        subspace of the reference batch.

    :param mnn_order: `list` or `None`, optional (default: None)
        The order in which batches are to be corrected. When set to None, datas are corrected
        sequentially.

    :param svd_mode: `str`, optional (default: 'rsvd')
        One of 'svd', 'rsvd', and 'irlb'. 'svd' computes SVD using a non-randomized SVD-via-ID
        algorithm, while 'rsvd' uses a randomized version. 'irlb' performes truncated SVD by
        implicitly restarted Lanczos bidiagonalization (forked from https://github.com/airysen/irlbpy).

    :param do_concatenate: `bool`, optional (default: True)
        Whether to concatenate the corrected matrices or AnnData objects. Default is True.

    :param save_raw: `bool`, optional (default: False)
        Whether to save the original expression data in the .raw attribute of AnnData objects.

    :param n_jobs: `int` or `None`, optional (default: None)
        The number of jobs. When set to None, automatically uses the number of cores.

    :param kwargs: `dict` or `None`, optional (default: None)
        optional keyword arguments for irlb.

    :return:
        datas: `numpy.ndarray` or class:`anndata.AnnData`
            Corrected matrix/matrices or AnnData object/objects, depending on the input type and
            do_concatenate.

        mnn_list_: `list`
            A list containing MNN pairing information as DataFrames in each iteration step.

        angle_list_: `list`
            A list containing angles of each batch.

    """
    if len(datas) < 2:
        return datas
    n_batch = len(datas)
    if mnn_order is not None:
        if sorted(mnn_order) != list(range(n_batch)):
            raise ValueError('The argument mnn_order should contain values in 1:' + 'n_batch' + '.')
    if isinstance(datas[0], AnnData):
        if var_index is not None:
            print('Inputs are AnnData objects, var_index ignored.')
        n_batch = len(datas)
        adata_vars = datas[0].var.index
        for i in range(1, n_batch):
            if (datas[i].var.index != adata_vars).any():
                raise ValueError('The AnnData objects have inconsistent number of vars.')
        if var_subset is not None and set(adata_vars) == set(var_subset):
            var_subset = None
        corrected = mnn_correct(*(adata.X for adata in datas), var_index=adata_vars,
                                var_subset=var_subset, k=k, sigma=sigma, cos_norm_in=cos_norm_in,
                                cos_norm_out=cos_norm_out, svd_dim=svd_dim, var_adj=var_adj,
                                compute_angle=compute_angle, mnn_order=mnn_order,
                                svd_mode=svd_mode, do_concatenate=do_concatenate, **kwargs)
        print('Packing AnnData object...')
        if do_concatenate:
            adata = AnnData.concatenate(*datas, batch_key=batch_key,
                                        batch_categories=batch_categories,
                                        index_unique=index_unique)
            if save_raw:
                adata.raw = adata.copy()
            adata.X = corrected[0]
            print('Done.')
            return adata, corrected[1], corrected[2]
        else:
            for adata, new_matrix in zip(datas, corrected[0]):
                if save_raw:
                    adata.raw = adata.copy()
                adata.X = new_matrix
            print('Done.')
            return datas, corrected[1], corrected[2]
    # ------------------------------------------------------------
    if n_jobs is None:
        n_jobs = cpu_count()
    n_cols = datas[0].shape[1]
    if len(var_index) != n_cols:
        raise ValueError('The number of vars is not equal to the length of var_index.')
    for i in range(1, n_batch):
        if datas[i].shape[1] != n_cols:
            raise ValueError('The input matrices have inconsistent number of columns.')
    # ------------------------------------------------------------
    print('Performing cosine normalization...')
    in_batches, out_batches, var_subset, same_set = transform_input_data(datas, cos_norm_in,
                                                                         cos_norm_out, var_index,
                                                                         var_subset, n_jobs)
    if mnn_order is None:
        mnn_order = list(range(n_batch))
    ref = mnn_order[0]
    ref_batch_in = in_batches[ref]
    if not same_set:
        ref_batch_out = out_batches[ref]
    res_container = [out_batches[ref]]
    mnn_container = [0]
    angle_container = [0]
    original_batch = [ref] * ref_batch_in.shape[0]
    print('Starting MNN correct iteration. Reference batch: ' + str(ref))
    # ------------------------------------------------------------
    # loop through batches
    for step in range(1, n_batch):
        target = mnn_order[step]
        print('Step ' + str(step) + ' of ' + str(n_batch - 1) + ': processing batch ' + str(target))
        new_batch_in = in_batches[target]
        if not same_set:
            new_batch_out = out_batches[target]
        print('  Looking for MNNs...')
        mnn_ref, mnn_new = find_mutual_nn(data1=ref_batch_in, data2=new_batch_in, k1=k, k2=k,
                                          n_jobs=n_jobs)
        print('  Computing correction vectors...')
        correction_in = compute_correction(ref_batch_in, new_batch_in, mnn_ref, mnn_new,
                                           new_batch_in, sigma)
        if not same_set:
            correction_out = compute_correction(ref_batch_out, new_batch_out, mnn_ref, mnn_new,
                                                new_batch_in, sigma)
        if compute_angle:
            print('  Computing angle...')
            ref_centred = ref_batch_in - np.mean(ref_batch_in, axis=0)
            ref_basis = svd_internal(ref_centred.T, nu=2, svd_mode=svd_mode, **kwargs)
            find_subspace_job = partial(find_shared_subspace, mat1=ref_basis, mat2_vec=True)
            with Pool(n_jobs) as p_n:
                angle_out = p_n.map(find_subspace_job, correction_in)
            angle_container.append(angle_out)
        # ------------------------
        if svd_dim is not None and svd_dim != 0:
            print('  Removing components...')
            mnn_ref_u = np.unique(mnn_ref)
            mnn_new_u = np.unique(mnn_new)
            in_span_ref = get_bio_span(ref_batch_in[mnn_ref_u, :], ndim=svd_dim, svd_mode=svd_mode,
                                       **kwargs)
            in_span_new = get_bio_span(new_batch_in[mnn_new_u, :], ndim=svd_dim, svd_mode=svd_mode,
                                       **kwargs)
            correction_in = subtract_bio(in_span_ref, in_span_new, correction=correction_in)
            if not same_set:
                out_span_ref = get_bio_span(ref_batch_out[mnn_ref_u, :], ndim=svd_dim,
                                            svd_mode=svd_mode, var_subset=var_subset, **kwargs)
                out_span_new = get_bio_span(new_batch_out[mnn_new_u, :], ndim=svd_dim,
                                            svd_mode=svd_mode, var_subset=var_subset, **kwargs)
                correction_out = subtract_bio(out_span_ref, out_span_new, correction=correction_out,
                                              var_subset=var_subset)
        # ------------------------
        if var_adj:
            print('  Adjusting variance...')
            correction_in = adjust_shift_variance(ref_batch_in, new_batch_in, correction_in, sigma,
                                                  n_jobs)
            if not same_set:
                correction_out = adjust_shift_variance(ref_batch_out, new_batch_out, correction_out,
                                                       sigma, n_jobs, var_subset)
        # ------------------------
        print('  Applying correction...')
        new_batch_in = new_batch_in + correction_in
        ref_batch_in = np.concatenate((ref_batch_in, new_batch_in))
        if same_set:
            res_container.append(new_batch_in)
        else:
            new_batch_out = new_batch_out + correction_out
            ref_batch_out = np.concatenate((ref_batch_out, new_batch_out))
            res_container.append(new_batch_out)
        mnn_container.append(DataFrame(
            {'new cell': mnn_new,
             'ref cell': mnn_ref,
             'original batch': [original_batch[mnn] for mnn in mnn_ref]}))
        original_batch += [target] * new_batch_in.shape[0]
    print('MNN correction complete. Gathering output...')
    reflow_order = [0] * n_batch
    for i in range(n_batch):
        reflow_order[mnn_order[i]] = i
    results_ = [np.array(res_container[i]) for i in reflow_order]
    mnn_list_ = [mnn_container[i] for i in reflow_order]
    angle_list_ = [angle_container[i] for i in reflow_order] if compute_angle else None
    if do_concatenate:
        results_ = np.concatenate(tuple(results_))
    return results_, mnn_list_, angle_list_
