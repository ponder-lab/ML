# Witness for wala/ML#880: the numpy `.T` transpose attribute must reverse axes
# like the `transpose` method, not collapse rank. np.eye(2, 4).T is (4, 2); before
# the fix the attribute resolved to (4,).
import numpy as np


def consume(x):
    pass


def consume_round_trip(x):
    pass


e = np.eye(2, 4).astype(np.float32)
consume(e.T)

# Round trip: `.T` reads a value produced by ANOTHER generator (the `np.transpose`
# result), not a direct allocation, exercising the receiver read through a computed
# base. np.transpose((2, 4)) is (4, 2), and its `.T` is (2, 4). This is the shape
# the sbcnm driver's `permutation(eye.T).T` takes.
consume_round_trip(np.transpose(e).T)
