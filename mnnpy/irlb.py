"""
See
https://github.com/airysen/irlbpy
for full repo.
"""
import numpy as np
import scipy.sparse as sparse
import warnings

from numpy.fft import rfft, irfft
import numpy.linalg as nla


# Matrix-vector product wrapper
# A is a numpy 2d array or matrix, or a scipy matrix or sparse matrix.
# x is a numpy vector only.
# Compute A.dot(x) if t is False,  A.transpose().dot(x)  otherwise.

def multA(A, x, TP=False, L=None):
    if sparse.issparse(A) :
        # m = A.shape[0]
        # n = A.shape[1]
        if TP:
            return sparse.csr_matrix(x).dot(A).transpose().todense().A[:, 0]
        return A.dot(sparse.csr_matrix(x).transpose()).todense().A[:, 0]
    if TP:
        return x.dot(A)
    return A.dot(x)


def multS(s, v, L, TP=False):

    N = s.shape[0]
    vp = prepare_v(v, N, L, TP=TP)
    p = irfft(rfft(vp) * rfft(s))
    if not TP:
        return p[:L]
    return p[L - 1:]


def prepare_s(s, L=None):
    N = s.shape[0]
    if L is None:
        L = N // 2
    K = N - L + 1
    return np.roll(s, K - 1)


def prepare_v(v, N, L, TP=False):
    v = v.flatten()[::-1]
    K = N - L + 1
    if TP:
        lencheck = L
        if v.shape[0] != lencheck:
            raise VectorLengthException('Length of v must be  L (if transpose flag is True)')
        pw = K - 1
        v = np.pad(v, (pw, 0), mode='constant', constant_values=0)
    elif not TP:
        lencheck = N - L + 1
        if v.shape[0] != lencheck:
            raise VectorLengthException('Length of v must be N-K+1')
        pw = L - 1
        v = np.pad(v, (0, pw), mode='constant', constant_values=0)
    return v


def orthog(Y, X):
    """Orthogonalize a vector or matrix Y against the columns of the matrix X.
    This function requires that the column dimension of Y is less than X and
    that Y and X have the same number of rows.
    """
    dotY = multA(X, Y, TP=True)
    return (Y - multA(X, dotY))

# Simple utility function used to check linear dependencies during computation:


def invcheck(x):
    eps2 = 2 * np.finfo(np.float).eps
    if(x > eps2):
        x = 1 / x
    else:
        x = 0
        warnings.warn(
            "Ill-conditioning encountered, result accuracy may be poor")
    return(x)


def lanczos(A, nval, tol=0.0001, maxit=50, center=None, scale=None, L=None):
    """Estimate a few of the largest singular values and corresponding singular
    vectors of matrix using the implicitly restarted Lanczos bidiagonalization
    method of Baglama and Reichel, see:

    Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
    J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005

    Keyword arguments:
    tol   -- An estimation tolerance. Smaller means more accurate estimates.
    maxit -- Maximum number of Lanczos iterations allowed.

    Given an input matrix A of dimension j * k, and an input desired number
    of singular values n, the function returns a tuple X with five entries:

    X[0] A j * nu matrix of estimated left singular vectors.
    X[1] A vector of length nu of estimated singular values.
    X[2] A k * nu matrix of estimated right singular vectors.
    X[3] The number of Lanczos iterations run.
    X[4] The number of matrix-vector products run.

    The algorithm estimates the truncated singular value decomposition:
    A.dot(X[2]) = X[0]*X[1].
    """
    mmult = None
    m = None
    n = None
    if A.ndim == 2:
        mmult = multA
        m = A.shape[0]
        n = A.shape[1]
        if(min(m, n) < 2):
            raise MatrixShapeException("The input matrix must be at least 2x2.")

    elif A.ndim == 1:
        mmult = multS
        A = np.pad(A, (0, A.shape[0] % 2), mode='edge')
        N = A.shape[0]
        if L is None:
            L = N // 2
        K = N - L + 1
        m = L
        n = K
        A = prepare_s(A, L)
    elif A.ndim > 2:
        raise MatrixShapeException("The input matrix must be 2D array")
    nu = nval

    m_b = min((nu + 20, 3 * nu, n))  # Working dimension size
    mprod = 0
    it = 0
    j = 0
    k = nu
    smax = 1
    # sparse = sparse.issparse(A)

    V = np.zeros((n, m_b))
    W = np.zeros((m, m_b))
    F = np.zeros((n, 1))
    B = np.zeros((m_b, m_b))

    V[:, 0] = np.random.randn(n)  # Initial vector
    V[:, 0] = V[:, 0] / np.linalg.norm(V)

    while it < maxit:
        if(it > 0):
            j = k

        VJ = V[:, j]

        # apply scaling
        if scale is not None:
            VJ = VJ / scale

        W[:, j] = mmult(A, VJ, L=L)
        mprod = mprod + 1

        # apply centering
        # R code: W[, j_w] <- W[, j_w] - ds * drop(cross(dv, VJ)) * du
        if center is not None:
            W[:, j] = W[:, j] - np.dot(center, VJ)

        if(it > 0):
            # NB W[:,0:j] selects columns 0,1,...,j-1
            W[:, j] = orthog(W[:, j], W[:, 0:j])
        s = np.linalg.norm(W[:, j])
        sinv = invcheck(s)
        W[:, j] = sinv * W[:, j]

        # Lanczos process
        while(j < m_b):
            F = mmult(A, W[:, j], TP=True, L=L)
            mprod = mprod + 1

            # apply scaling
            if scale is not None:
                F = F / scale

            F = F - s * V[:, j]
            F = orthog(F, V[:, 0:j + 1])
            fn = np.linalg.norm(F)
            fninv = invcheck(fn)
            F = fninv * F
            if(j < m_b - 1):
                V[:, j + 1] = F
                B[j, j] = s
                B[j, j + 1] = fn
                VJp1 = V[:, j + 1]

                # apply scaling
                if scale is not None:
                    VJp1 = VJp1 / scale

                W[:, j + 1] = mmult(A, VJp1, L=L)
                mprod = mprod + 1

                # apply centering
                # R code: W[, jp1_w] <- W[, jp1_w] - ds * drop(cross(dv, VJP1))
                # * du
                if center is not None:
                    W[:, j + 1] = W[:, j + 1] - np.dot(center, VJp1)

                # One step of classical Gram-Schmidt...
                W[:, j + 1] = W[:, j + 1] - fn * W[:, j]
                # ...with full reorthogonalization
                W[:, j + 1] = orthog(W[:, j + 1], W[:, 0:(j + 1)])
                s = np.linalg.norm(W[:, j + 1])
                sinv = invcheck(s)
                W[:, j + 1] = sinv * W[:, j + 1]
            else:
                B[j, j] = s
            j = j + 1
        # End of Lanczos process
        S = nla.svd(B)
        R = fn * S[0][m_b - 1, :]  # Residuals
        if it == 0:
            smax = S[1][0]  # Largest Ritz value
        else:
            smax = max((S[1][0], smax))

        conv = sum(np.abs(R[0:nu]) < tol * smax)
        if(conv < nu):  # Not coverged yet
            k = max(conv + nu, k)
            k = min(k, m_b - 3)
        else:
            break
        # Update the Ritz vectors
        V[:, 0:k] = V[:, 0:m_b].dot(S[2].transpose()[:, 0:k])
        V[:, k] = F
        B = np.zeros((m_b, m_b))
        # Improve this! There must be better way to assign diagonal...
        for l in range(k):
            B[l, l] = S[1][l]
        B[0:k, k] = R[0:k]
        # Update the left approximate singular vectors
        W[:, 0:k] = W[:, 0:m_b].dot(S[0][:, 0:k])
        it = it + 1

    U = W[:, 0:m_b].dot(S[0][:, 0:nu])
    V = V[:, 0:m_b].dot(S[2].transpose()[:, 0:nu])
    # return((U, S[1][0:nu], V, it, mprod))

    return LanczosResult(**{'U': U,
                            's': S[1][0:nu],
                            'V': V,
                            'steps': it,
                            'nmult': mprod
                            })


class LanczosResult():

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])


class VectorLengthException(Exception):
    pass


class MatrixShapeException(Exception):
    pass
