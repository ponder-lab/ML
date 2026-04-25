import json

import tensorflow as tf


def f(a):
    pass


# `json.loads` is not modeled by the static analyzer — its return value has empty PTS, so the
# resulting `tf.keras.Input(shape=...)` call's `shape` argument is unresolvable from Ariadne's
# perspective. This drives `Input.getShapes` through the default-shape path that wala/ML#355 fixed.
#
# Runtime: `arg.shape == (None, 32)` and `arg.dtype == tf.float32`. Analyzer expectation
# (post-#355): the call result is recognized as a tensor with ⊤ shape (`null` dims) and known
# `float32` dtype — i.e. `TensorType(float32, null)`, NOT ⊥ (which would silently drop the
# variable from downstream analysis).
shape = json.loads("[32]")
arg = tf.keras.Input(shape=shape)
assert arg.shape == (None, 32)
assert arg.dtype == tf.float32

f(arg)
