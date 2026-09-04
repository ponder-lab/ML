# Witnesses for wala/ML#866: NumPy fancy indexing replaces the receiver's
# leading axis with the index's shape rather than peeling it. The scalar
# spelling produces the same OUTPUT as peeling on a rank-2 receiver and is
# correct there, which is the contrast that makes a wrong fix visible: a fix
# that merely stopped peeling would break it.
import numpy as np


def eye_fancy():
    return np.eye(3)[np.array([0, 1, 2])]


def eye_scalar():
    return np.eye(3)[0]


def ones3_bare():
    return np.ones([2, 3, 4])


def ones3_fancy():
    return np.ones([2, 3, 4])[np.array([0, 1])]


def ones2_fancy():
    return np.ones([3, 3])[np.array([0, 1])]


def mask_fancy():
    # A boolean index is a mask whose selected count is a runtime fact: the
    # analysis keeps the receiver's rank with an unresolved leading axis.
    return np.eye(3)[np.array([True, False, True])]


assert eye_fancy().shape == (3, 3)
assert eye_scalar().shape == (3,)
assert ones3_bare().shape == (2, 3, 4)
assert ones3_fancy().shape == (2, 3, 4)
assert ones2_fancy().shape == (2, 3)
assert mask_fancy().shape == (2, 3)
