import os

import tensorflow as tf


def consume(t):
    pass


# Evidence-free twin of `tf2_test_reshape_tf_shape.py` (wala/ML#722): the input's
# leading axes are fixed runtime sizes the analysis cannot compute (environment
# reads), so `tf.shape(x)[0]`/`[1]` carry *no* `None`-evidence and the reshape
# target's leading elements classify `Unresolved` per the wala/ML#721 criterion,
# while the trailing constant stays exact. Contrast the sibling fixture, where
# the batch axis is declared `None` and the same pattern yields `Dynamic`.
n = int(os.environ.get("ARIADNE_TEST_N", "2"))
m = int(os.environ.get("ARIADNE_TEST_M", "4"))

x = tf.ones((n, m, 6))

batch = tf.shape(x)[0]
seq = tf.shape(x)[1]

h = tf.reshape(x, [-1, 6])
w = tf.ones((6, 10))
out = tf.matmul(h, w)
out = tf.reshape(out, [batch, seq, 10])

assert out.shape == (2, 4, 10)
assert out.dtype == tf.float32

consume(out)
