import os

import tensorflow as tf


def consume(t):
    pass


# Mixed reshape target (wala/ML#741): an `Unresolved` leading element (an
# environment-read size the analysis cannot compute, per the wala/ML#721
# criterion) alongside the literal `-1` placeholder. The placeholder must
# surface as the symbolic unknown-size dimension, not as a fixed size of -1:
# the follow-on addition broadcasts the reshaped tensor against a fully
# concrete one, which a raw -1 "size" would wrongly reject.
n = int(os.environ.get("ARIADNE_TEST_N", "2"))

x = tf.ones((n, 3, 8))

batch = tf.shape(x)[0]

y = tf.reshape(x, [batch, -1, 8])
w = tf.ones((2, 3, 8))
z = y + w

assert z.shape == (2, 3, 8)
assert z.dtype == tf.float32

consume(z)
