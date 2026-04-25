import tensorflow as tf
import numpy as np


def consume(tensor):
    pass


# Regression guard for wala/ML#403: chained `astype` on an mnist receiver. The first cast's result
# is a synthetic-method return whose PointerKey is implicit, so the receiver lookup for the second
# `astype` call exercises the IAE-catch fallback path in `AstypeOperation.getDefaultShapes`.
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_train.dtype == np.uint8

y = x_train.astype(np.int32).astype(np.float32)
assert y.shape == (60000, 28, 28)
assert y.dtype == np.float32

consume(y)
