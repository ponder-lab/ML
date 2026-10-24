# Test `tf.ensure_shape` over a tensor that only the sidecar types: the declared shape's `None`
# keeps the input's extent, so the result is (2, 3) float32, the input's shape refined by the
# declaration, never (2, None).
import os
import pickle
import tempfile

import tensorflow as tf
from loader import load_ensured


def consume_ensured(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


path = os.path.join(tempfile.mkdtemp(), "tensor.pickle")
with open(path, "wb") as f:
    pickle.dump(tf.ones([2, 3]), f)
consume_ensured(load_ensured(path))
