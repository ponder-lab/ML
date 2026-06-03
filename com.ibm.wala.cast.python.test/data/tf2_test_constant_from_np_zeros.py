# Companion to `tf2_test_constant_from_numpy.py` for the `np.zeros → tf.constant`
# bridge (wala/ML#539). Exercises the `NpZeros` manual-generator recovery path
# (`createManualGenerator`) — symmetric to the `np.ones` bridge but with a
# non-float dtype — so the `tf.constant` result is `(2, 3) int32` rather than ⊤.
import numpy as np
import tensorflow as tf


def consume(x):
    """Sink function — pins the dtype/shape of whatever flows in as `x`."""
    return x


arr_from_np = tf.constant(np.zeros((2, 3), dtype=np.int32))
assert arr_from_np.shape == (2, 3)
assert arr_from_np.dtype == tf.int32
consume(arr_from_np)
