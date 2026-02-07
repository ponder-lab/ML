import tensorflow as tf


def test(t1, t2):
    pass


# Positional (shape, alpha) and keyword (beta, dtype)
# shape=(2,), alpha=1.0 (scalar). Result shape: (2,).
t1 = tf.random.gamma((2,), 1.0, beta=2.0, dtype=tf.float64)
assert t1.shape == (2,)
assert t1.dtype == tf.float64

# Keyword args
# shape=(2,), alpha=[1.0, 2.0] (shape (2,)). Result shape: (2, 2).
t2 = tf.random.gamma(shape=(2,), alpha=[1.0, 2.0])
assert t2.shape == (2, 2)
assert t2.dtype == tf.float32

test(t1, t2)
