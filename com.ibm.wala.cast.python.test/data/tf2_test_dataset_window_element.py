# Test https://github.com/wala/ML/issues/164: `window` yields datasets whose own iteration yields
# the source elements, and for a tuple element a tuple of datasets, one per component.
import tensorflow as tf


def consume_window(w):
    pass


def consume_row(e):
    pass


def consume_first_window(a):
    pass


def consume_first_row(x):
    pass


def consume_second_row(y):
    pass


for w in tf.data.Dataset.from_tensor_slices(tf.ones([8, 4])).window(2):
    assert isinstance(w, tf.data.Dataset)
    consume_window(w)
    for e in w:
        assert e.shape == (4,) and e.dtype == tf.float32
        consume_row(e)

pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)
for a, b in pairs.window(2):
    assert isinstance(a, tf.data.Dataset) and isinstance(b, tf.data.Dataset)
    consume_first_window(a)
    for x in a:
        assert x.shape == (4,) and x.dtype == tf.float32
        consume_first_row(x)
    for y in b:
        assert y.shape == (3,) and y.dtype == tf.int32
        consume_second_row(y)
