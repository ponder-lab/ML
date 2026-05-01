import tensorflow as tf
import numpy as np


def consume(tensor):
    pass


# `x_train` is a (60000, 28, 28) uint8 ndarray from mnist. Function-style
# `np.array(x, dtype)` should preserve the input's shape while applying the new
# dtype, mirroring `ndarray.astype`'s shape-preserving / dtype-changing semantics.
# As of the time this fixture was added, `np.array(...)` is modeled in
# `tensorflow.xml` as a fresh `Ltensorflow/python/framework/ops/ndarray`
# allocation with no shape transfer from the input, so the static analysis
# reports {? unknown} for `y`. This fixture is the positive regression guard for
# the fix that will preserve shape through `np.array(x, dtype)`.
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_train.dtype == np.uint8

y = np.array(x_train, np.float32)
assert y.shape == (60000, 28, 28)
assert y.dtype == np.float32

consume(y)
