import json

import tensorflow as tf


def f(a):
    pass


# Regression vehicle for wala/ML#355: drives `Input.getShapes` through the unresolvable-shape
# default path. `json.loads` is intentionally chosen because Ariadne does not model it (its
# return-value PTS is empty), so the static analyzer cannot resolve the `shape` argument here even
# though the runtime trivially can.
#
# The asymmetry between the runtime shape `(None, 32)` and the JUnit expectation
# `TENSOR_UNKNOWN_SHAPE_FLOAT32` (⊤-dims, float32) is therefore *deliberate* — it documents the
# graceful-degradation property of the lattice: when one axis (here, shape) is unresolvable, the
# orthogonal axis (dtype) still carries `float32` rather than collapsing to ⊥. wala/ML#355 was
# specifically about ensuring `Input.getDefaultShapes` returns ⊤ in this case, not ⊥.
#
# Both asserts are kept (and run): the shape assert documents the runtime truth so a future
# reader can gauge how much the analyzer is leaving on the table; the dtype assert is the part
# that the JUnit expectation matches.
shape = json.loads("[32]")
arg = tf.keras.Input(shape=shape)
assert arg.shape == (None, 32)
assert arg.dtype == tf.float32

f(arg)
