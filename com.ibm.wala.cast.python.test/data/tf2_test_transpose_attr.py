# Witness for wala/ML#880: the numpy `.T` transpose attribute must reverse axes
# like the `transpose` method, not collapse rank. np.eye(2, 4).T is (4, 2); before
# the fix the attribute resolved to (4,).
import numpy as np


def consume(x):
    pass


e = np.eye(2, 4).astype(np.float32)
consume(e.T)
