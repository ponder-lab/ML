import json

import tensorflow as tf


def f(a):
    pass


# `json.loads` is not modeled by the static analyzer — its return value has empty PTS, so the
# resulting `tf.keras.Input(shape=...)` call's `shape` argument is unresolvable from Ariadne's
# perspective. This drives `Input.getShapes` through the default-shape path that wala/ML#355 fixed.
#
# The static-analysis expectation is `TensorType(float32, null)` — a tensor with concrete dtype
# and ⊤ shape, NOT ⊥ (which would silently drop the variable from downstream analysis). We
# deliberately do NOT assert `arg.shape` here: the runtime shape is `(None, 32)`, but the analyzer
# legitimately cannot recover this (the whole point of the test is the unresolvable-shape path),
# so a runtime shape assert would mismatch the JUnit expectation. Only `dtype` is asserted because
# both runtime and analyzer agree there.
shape = json.loads("[32]")
arg = tf.keras.Input(shape=shape)
assert arg.dtype == tf.float32

f(arg)
