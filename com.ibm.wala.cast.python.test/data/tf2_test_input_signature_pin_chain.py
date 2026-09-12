# Witness for wala/ML#810's safety residual. A `tf.function` `input_signature` declares its
# parameter's rank, so an analysis that reads the declaration can pin the parameter of `g1` from
# `(None, 4)` when its argument is opaque (a list built in a loop). That pinned value is then the
# ARGUMENT of `g2` and `g3`. Inside a decorated body TensorFlow relaxes the parameter's static shape to
# the signature, so `g2`, declaring the looser `(None, None)`, sees `(None, None)` even though `g1`
# handed it a value that `g1` sees as `(None, 4)`; `g3`, declaring the same `(None, 4)`, sees `(None, 4)`.
# The concrete value is `(3, 4)` float32 throughout. The static shapes are asserted inside each body.
import tensorflow as tf

sig_tight = [tf.TensorSpec(shape=(None, 4), dtype=tf.float32)]
sig_loose = [tf.TensorSpec(shape=(None, None), dtype=tf.float32)]


def consume_first(a):
    pass


def consume_looser(a):
    pass


def consume_same(a):
    pass


@tf.function(input_signature=sig_loose)
def g2(z):
    assert z.shape.as_list() == [None, None]
    assert z.dtype == tf.float32
    consume_looser(z)
    return z


@tf.function(input_signature=sig_tight)
def g3(z):
    assert z.shape.as_list() == [None, 4]
    assert z.dtype == tf.float32
    consume_same(z)
    return z


@tf.function(input_signature=sig_tight)
def g1(y):
    assert y.shape.as_list() == [None, 4]
    assert y.dtype == tf.float32
    consume_first(y)
    g2(y)
    g3(y)
    return y


rows = []
for i in range(3):
    rows.append([float(i)] * 4)
y = tf.constant(rows, dtype=tf.float32)
assert y.shape == (3, 4)
assert y.dtype == tf.float32
out = g1(y)
assert out.shape == (3, 4)
