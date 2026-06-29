import tensorflow as tf


def consume(x):
    pass


# wala/ML#637's example verbatim: a `complex64` constant built from complex literals. The front end
# skips constant-folding the `2j` literal (wala/ML#642) instead of aborting the module, so this
# builds and types normally; the companion `tf2_test_complex64.py` uses integer values to isolate
# the dtype modeling.
z = tf.constant([1 + 2j, 3 + 4j], dtype=tf.complex64)
assert z.shape == (2,)
assert z.dtype == tf.complex64
consume(z)
