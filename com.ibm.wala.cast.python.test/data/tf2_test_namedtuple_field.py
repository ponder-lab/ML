from typing import NamedTuple, List

import numpy as np
import tensorflow as tf


# Documents the wala/ML#579 gap: a tensor stored in a user-defined `NamedTuple`
# field and read back out loses its tensor type. At runtime `b` is the original
# `(4, 8) float32` tensor (the asserts below hold), but the static analysis
# currently recovers `consume`'s parameter as *no* tensor at all. This is the
# concrete blocker for typing the GCN layer outputs in wala/ML#570, where
# `GraphConvolution.call` unwraps a `GNNInput` `NamedTuple` the same way.
class Wrapper(NamedTuple):
    tensor: tf.Tensor
    rest: List


def consume(x):
    pass


a = tf.constant(np.ones((4, 8), dtype=np.float32))
assert a.shape == (4, 8) and a.dtype == tf.float32
w = Wrapper(a, [])
b = w.tensor
assert b.shape == (4, 8) and b.dtype == tf.float32
consume(b)
