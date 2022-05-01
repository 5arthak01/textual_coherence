import numpy as np


def random_permutation_matrix(N, dtype=np.float32):
    """
    Generate a random permutation matrix.

    :param N: dimension of the permutation matrix
    :return: a numpy array with shape (N, N)
    """
    A = np.identity(N, dtype=dtype)
    idx = np.random.permutation(N)
    return A[idx, :]
