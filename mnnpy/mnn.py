from os import cpu_count
from multiprocessing import Pool
from functools import partial
import numpy as np
from anndata import AnnData
from pandas import DataFrame
import pandas as pd
from .utils import transform_input_data, compute_correction
from .utils import svd_internal, find_shared_subspace, get_bio_span, subtract_bio
from .utils import adjust_shift_variance, l2_norm, scale_rows
from scipy.sparse import issparse
from scipy.spatial import cKDTree

###############################################################################
##################### Marioni MNN Alignment ###################################
###############################################################################

def marioniCorrect(ref_mat, targ_mat, k1=20, k2=20, fk=5, ndist=3, var_index=None, var_subset=None, 
                   cosine_norm=True, n_jobs=None):
  """marioniCorrect is a function that corrects for batch effects using the Marioni method.

  Args:
      ref_mat (pd.Dataframe): matrix of samples by genes of cPC corrected data that serves as the reference data in the MNN alignment.
          In the standard Celligner pipeline this the cell line data.
      targ_mat matrix of samples by genes of cPC corrected data that is corrected in the MNN alignment and projected onto the reference data.
          In the standard Celligner pipeline this the tumor data.
      mnn_kwargs (dict): args to mnnCorrect
      k1 (int): number of nearest neighbors to use for the first batch
      k2 (int): number of nearest neighbors to use for the second batch
      fk (int): number of nearest neighbors to use for the first batch
      ndist (int): number of nearest neighbors to use for the first batch
      var_index (pd.Series): index of variables to use for the first batch
      var_subset (list): list of variables to use for the first batch
      n_jobs (int): number of jobs to use for parallelization

  Returns:
      pd.Dataframe: corrected dataframe
  """
  if n_jobs is None:
      n_jobs = cpu_count()
  n_cols = ref_mat.shape[1]
  if len(var_index) != n_cols:
      raise ValueError('The number of vars is not equal to the length of var_index.')
  if targ_mat.shape[1] != n_cols:
      raise ValueError('The input matrices have inconsistent number of columns.')
  
  if var_subset is not None:
      subref_mat = ref_mat.loc[:, var_subset].values
      subtarg_mat = targ_mat.loc[:, var_subset].values
  else:
      subref_mat = ref_mat.values
      subtarg_mat = targ_mat.values

  if cosine_norm:
      print('Performing cosine normalization...')
      in_batches = _cosineNormalization(subref_mat, subtarg_mat, 
          cos_norm_in=True, cos_norm_out=True, n_jobs=n_jobs)
      subref_mat, subtarg_mat = in_batches
      del in_batches
      #in_batches = _cosineNormalization(ref_mat, targ_mat, 
      #    cos_norm_in=True, cos_norm_out=True, n_jobs=n_jobs)
      #ref_mat, targ_mat = in_batches
      #del in_batches
  
  print('  Looking for MNNs...')
  mnn_pairs = findMutualNN(data1=subref_mat, data2=subtarg_mat, k1=k1, k2=k2, n_jobs=n_jobs)
  print('  Found '+str(len(mnn_pairs))+' mutual nearest neighbors.')
  mnn_ref, mnn_targ = np.array(mnn_pairs).T
  
  # TODO: this block shouldn't be usefull
  idx=np.argsort(mnn_ref)
  mnn_ref=mnn_ref[idx]
  mnn_targ=mnn_targ[idx]
  #import ipdb; ipdb.set_trace()

  # compute the overall batch vector
  corvec, _ = _averageCorrection(ref_mat.values, mnn_ref, targ_mat.values, mnn_targ)
  overall_batch = corvec.mean(axis=0)

  # remove variation along the overall batch vector
  ref_mat = _squashOnBatchDirection(ref_mat.values, overall_batch)
  targ = _squashOnBatchDirection(targ_mat.values, overall_batch)
  # recompute correction vectors and apply them
  re_ave_out, npairs = _averageCorrection(ref_mat, mnn_ref, targ, mnn_targ)
  del subref_mat, subtarg_mat, ref_mat
  # TODO: why cKDTRee results depend on how we order the input matrix' datapoints??
  distances, index = cKDTree(np.take(targ, np.sort(npairs), 0)[:,var_subset]).query(
      x=targ[:, var_subset],
      k=min(fk, len(npairs)),
      n_jobs=n_jobs)
  targ_mat = pd.DataFrame(data=targ, columns=targ_mat.columns, index=targ_mat.index)
  targ_mat += _computeTricubeWeightedAvg(re_ave_out[np.argsort(npairs)], index, distances, ndist=ndist)
  return targ_mat, mnn_pairs

#@jit((float32[:, :], float32[:, :], int8, int8, int8))
def findMutualNN(data1, data2, k1, k2, n_jobs):
  """findMutualNN finds the mutual nearest neighbors between two sets of data.

  Args:
      data1 ([type]): [description]
      data2 ([type]): [description]
      k1 ([type]): [description]
      k2 ([type]): [description]
      n_jobs ([type]): [description]

  Returns:
      [type]: [description]
  """
  k_index_1 = cKDTree(data1).query(x=data2, k=k1, n_jobs=n_jobs)[1]
  k_index_2 = cKDTree(data2).query(x=data1, k=k2, n_jobs=n_jobs)[1]
  mutuale = []
  for index_2, val in enumerate(k_index_1):
      for index_1 in val:
          if index_2 in k_index_2[index_1]:
              mutuale.append((index_1, index_2))
  return mutuale

def _cosineNormalization(*datas, cos_norm_in, cos_norm_out, n_jobs):
  """_cosineNormalization transforms input data to be centered and normalized.

  Args:
      cos_norm_in ([type]): [description]
      cos_norm_out ([type]): [description]
      n_jobs ([type]): [description]

  Returns:
      [type]: [description]
  """
  datas = [data.toarray().astype(np.float32) if issparse(data) else data.astype(np.float32) for data in datas]
  with Pool(n_jobs) as p_n:
      in_scaling = p_n.map(l2_norm, datas)
  in_scaling = [scaling[:, None] for scaling in in_scaling]
  if cos_norm_in:
      with Pool(n_jobs) as p_n:
          datas = p_n.starmap(scale_rows, zip(datas, in_scaling))
  return datas

def _averageCorrection(refdata, mnn1, curdata, mnn2):
  """_averageCorrection computes correction vectors for each MNN pair, and then averages them for each MNN-involved cell in the second batch.

  Args:
      refdata (pandas.DataFrame): matrix of samples by genes of cPC corrected data that serves as the reference data in the MNN alignment.
      mnn1 (list): mnn1 pairs
      curdata (pandas.DataFrame): matrix of samples by genes of cPC corrected data that is corrected in the MNN alignment and projected onto the reference data.
      mnn2 (list): mnn2 pairs

  Returns:
      dict: correction vector and pairs
  """
  npairs = pd.Series(mnn2).value_counts()
  corvec = np.take(refdata, mnn1, 0) - np.take(curdata, mnn2, 0)
  cor = np.zeros((len(npairs),corvec.shape[1]))
  mnn2 = np.array(mnn2)
  #mnn2_sort = np.sort(mnn_targ)
  for i, v in enumerate(set(mnn2)):
      cor[i] = corvec[mnn2==v].sum(0)/npairs[v]
  return cor, list(set(mnn2))

def _squashOnBatchDirection(mat, batch_vec):
  """_squashOnBatchDirection - Projecting along the batch vector, and shifting all samples to the center within each batch.

  Args:
      mat (pandas.DataFrame): matrix of samples by genes
      batch_vec (pandas.Series): batch vector

  Returns:
      pandas.DataFrame: corrected matrix
  """
  batch_vec = batch_vec/np.sqrt(np.sum(batch_vec**2))
  batch_loc = np.dot(mat, batch_vec)
  mat = mat + np.outer(np.mean(batch_loc) - batch_loc, batch_vec)
  return mat

def _computeTricubeWeightedAvg(vals, indices, distances, bandwidth=None, ndist=3):
  """_computeTricubeWeightedAvg - Centralized function to compute tricube averages.

  Args:
      vals (pandas.DataFrame): correction vector
      indices (pandas.DataFrame): nxk matrix for the nearest neighbor indice
      distances (pandas.DataFrame): nxk matrix for the nearest neighbor Euclidea distances
      bandwidth (float): Is set at 'ndist' times the median distance, if not specified.
      ndist (int, optional): By default is MNN_NDIST.

  Returns:
      [type]: [description]
  """
  if bandwidth is None:
      middle = int(np.floor(indices.shape[1]/2))
      mid_dist = distances[:,middle]
      bandwidth = mid_dist * ndist
  bandwidth = np.maximum(1e-8, bandwidth)

  rel_dist = distances.T/bandwidth
  # don't use pmin(), as this destroys dimensions.
  rel_dist[rel_dist > 1] = 1
  tricube = (1 - rel_dist**3)**3
  weight = tricube/np.sum(tricube, axis=0)
  del rel_dist, tricube, bandwidth
  #import ipdb; ipdb.set_trace()
  output = np.zeros((indices.shape[0], vals.shape[1]))
  for kdx in range(indices.shape[1]):
      output += np.einsum("ij...,i...->ij...", vals[indices[:,kdx]], weight[kdx])
  return output

###############################################################################
##################### Regular MNN Alignment ###################################
###############################################################################

def mnn_correct(*datas, var_index=None, var_subset=None, batch_key='batch', index_unique='-',
                batch_categories=None, k1=20, k2=20, sigma=1., cos_norm_in=True, cos_norm_out=True,
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
                                var_subset=var_subset, k1=k1, k2=k2, sigma=sigma, cos_norm_in=cos_norm_in,
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
        mnn = findMutualNN(data1=ref_batch_in, data2=new_batch_in, k1=k1, k2=k2,
                                          n_jobs=n_jobs)
        val = np.array(mnn)
        mnn_ref = val[:,0]
        mnn_new = val[:,1]
        print('found ' + str(len(mnn_ref)) + " mnns..")
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
