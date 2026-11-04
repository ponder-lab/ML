# Test https://github.com/wala/ML/issues/1211: a zip of datasets built by `from_tensors` reads each
# member of their elements at its own type.
import tensorflow as tf


def consume_unzipped_a(a):
    pass


def consume_zipped_a(a):
    pass


def consume_zipped_b(b):
    pass


def consume_zipped_c(c):
    pass


def consume_zipped_flat(f):
    pass


pair = tf.data.Dataset.from_tensors((tf.ones([2, 4]), tf.ones([2, 3])))
label = tf.data.Dataset.from_tensors(tf.zeros([2]))

for a, b in pair:
    assert a.shape == (2, 4) and a.dtype == tf.float32
    consume_unzipped_a(a)

for (a, b), c in tf.data.Dataset.zip((pair, label)):
    assert a.shape == (2, 4) and a.dtype == tf.float32
    assert b.shape == (2, 3) and b.dtype == tf.float32
    assert c.shape == (2,) and c.dtype == tf.float32
    consume_zipped_a(a)
    consume_zipped_b(b)
    consume_zipped_c(c)

rows = tf.data.Dataset.from_tensor_slices(tf.ones([1, 5], tf.int32))
for f, c in tf.data.Dataset.zip((rows, label)):
    assert f.shape == (5,) and f.dtype == tf.int32
    consume_zipped_flat(f)
