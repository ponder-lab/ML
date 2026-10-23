# Test https://github.com/wala/ML/issues/1008 and https://github.com/wala/ML/issues/865: `np.mean`
# over a value whose int64 dtype only the sidecar supplies, with a `dtype` argument the analysis
# cannot read. The result's dtype is the argument's, float32 at run time, so it must read as
# unknown, never as the float64 an int64 input averages in.
import os
import pickle
import tempfile

import numpy as np
from loader import load_mean


def consume_mean(m):
    assert np.ndim(m) == 0 and m.dtype == np.float32


path = os.path.join(tempfile.mkdtemp(), "seq.pickle")
with open(path, "wb") as f:
    pickle.dump([1, 2, 3, 4], f)
consume_mean(load_mean(path))
