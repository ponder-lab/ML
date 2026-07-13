import tensorflow as tf


def consume(t):
    pass


# tf.matmul's batched form (wala/ML#718): the leading (batch) dimensions carry
# through and the trailing two dimensions compose as the matrix product, so the
# rank is preserved; the analysis previously collapsed every product to rank
# two.
a = tf.ones((2, 4, 3))
b = tf.ones((2, 3, 5))
out = tf.matmul(a, b)

assert out.shape == (2, 4, 5)
assert out.dtype == tf.float32

consume(out)
