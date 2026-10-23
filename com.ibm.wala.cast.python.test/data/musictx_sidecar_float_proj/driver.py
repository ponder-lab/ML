# Test https://github.com/wala/ML/issues/1008: a batch padded by `np.append(data, 0.5)` is
# `float64` at run time, where the sidecar types the unpickled sequence `int64`. The padding arm's
# dtype is the promotion of the two, so the batch must not read as `int64` alone.
import numpy as np
from data import Data


def consume_padded_batch(b):
    assert b.shape == (2, 6) and b.dtype == np.float64


dataset = Data(2, 4)
consume_padded_batch(dataset.batch(2, 6))
