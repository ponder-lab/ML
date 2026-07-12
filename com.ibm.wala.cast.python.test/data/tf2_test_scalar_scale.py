import math

import tensorflow as tf


def consume(t):
    pass


# A tensor scaled by a statically opaque scalar expression (wala/ML#718): the
# expression's value never resolves (the analysis does not fold arithmetic),
# but its scalarness is structural, so the broadcast preserves the tensor's
# shape; the analysis previously erased the result to "not a tensor."
t = tf.ones((2, 4))
out = t * (1.0 / math.sqrt(4.0))

assert out.shape == (2, 4)
assert out.dtype == tf.float32

consume(out)
