# `np.unique` in its flag form (wala/ML#799): the inverse output is always int64 regardless
# of the input dtype, the values output preserves the input dtype, and the corpus's two-branch
# label idiom must yield a dtype union rather than a wrong definite.
import numpy as np


def consume(a):
    pass


def consume2(b):
    pass


def consume3(c):
    pass


labels = np.array([3, 1, 3, 2], dtype=int)
_, _, inv = np.unique(labels, return_index=True, return_inverse=True)
assert inv.dtype == np.int64
assert inv.shape == (4,)
consume(inv)

vals = np.unique(labels, return_index=True, return_inverse=True)[0]
assert vals.dtype == np.int64
assert vals.shape == (3,)
consume2(vals)


# The TUDataset.read_data two-branch idiom: regression labels are float, classification labels
# are int densified through the unique inverse; the union must carry both.
def read(flag):
    if flag:
        y = np.array([1.5, 2.5], dtype=float)
    else:
        y = np.array([3, 1, 3], dtype=int)
        _, _, y = np.unique(y, return_index=True, return_inverse=True)
        y = np.reshape(y, y.shape)
    return y


ya = read(True)
assert ya.dtype == np.float64
consume3(ya)
yb = read(False)
assert yb.dtype == np.int64
consume3(yb)
