import numpy as np
import tensorflow as tf


def consume(tensor):
    pass


# A slice of a slice. The inner slice `x_train[:5]` recovers `(5, 28, 28) uint8`
# because its receiver is the mnist source. The outer slice's receiver is the
# inner slice's *result*, whose dtype the PTS walk can't see (empty PTS) and the
# SSA-DU fallback misses unless it recurses through the dtype-preserving slice op.
# This fixture pins the outer slice's dtype as the regression guard for that
# recovery (wala/ML#602).
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_train.dtype == np.uint8

y = x_train[:5][:3]
assert y.shape == (3, 28, 28)
assert y.dtype == np.uint8

consume(y)
