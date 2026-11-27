# Test https://github.com/wala/ML/issues/1344.
import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def sp_matrix_to_sp_tensor(x):
    x = x.tocoo()
    indices = np.array([x.row, x.col]).T
    return tf.SparseTensor(indices, x.data, x.shape)


def f(a):
    assert isinstance(a, tf.SparseTensor)
    assert a.shape == (6, 6)
    assert a.dtype == tf.float32


a = sp.block_diag([sp.csr_matrix(np.eye(3, dtype=np.float32))] * 2)
f(sp_matrix_to_sp_tensor(a))
