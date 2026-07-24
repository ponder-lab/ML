# Test the scipy.sparse matrix product (wala/ML#766). The row-normalization idiom from NLPGNN's
# Planetoid loader: the product of a SciPy sparse matrix and a dense float32 array is a dense
# float32 array.

import numpy as np
import scipy.sparse as sp


def consume(features):
    pass


features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
rowsum = np.array(features.sum(1))
r_inv = np.power(rowsum, -1).flatten()
r_mat_inv = sp.diags(r_inv)
features = r_mat_inv.dot(features)
assert features.shape == (2, 2)
assert features.dtype == np.float32
consume(features)
