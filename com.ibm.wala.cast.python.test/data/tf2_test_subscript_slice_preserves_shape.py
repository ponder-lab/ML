import tensorflow as tf
import numpy as np


def consume(tensor):
    pass


# A constant-step slice `x[:k]` on an ndarray should replace the leading
# dimension with `k` and keep the remaining dimensions intact. This fixture
# is the positive regression guard for the analyzer-side fix: currently
# `NdarraySubscriptOperation` doesn't propagate the receiver's shape through
# a constant slice, so the sliced tensor registers as {? unknown} in the
# static analysis. Fixing this closes the last remaining gap in
# `testNeuralNetwork` — its union expected the (5, 784) shape produced by
# `x_test[:n_images]` at the visualization call site in `neural_network.py`.
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_train.dtype == np.uint8

y = x_train[:5]
assert y.shape == (5, 28, 28)
assert y.dtype == np.uint8

consume(y)
