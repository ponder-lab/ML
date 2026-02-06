import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def h(a):
    pass


def i(a):
    pass


def j(a):
    pass


# tf.constant keyword args
x = tf.constant(value=[[1.0, 1.0], [2.0, 2.0]], dtype=tf.float32)
# Check x shape? tf.constant result is not usually asserted in these tests, but used.

# 1. Keyword args
r1 = tf.reduce_mean(input_tensor=x, axis=0)
assert r1.shape == (2,)
assert r1.dtype == tf.float32
f(r1)

# 2. Mixed (pos + kw)
r2 = tf.reduce_mean(x, axis=1)
assert r2.shape == (2,)
assert r2.dtype == tf.float32
g(r2)

# 3. Mixed (pos + kw) with keepdims
r3 = tf.reduce_mean(x, 0, keepdims=True)
assert r3.shape == (1, 2)
assert r3.dtype == tf.float32
h(r3)

# 4. tf.math.reduce_mean keyword args
r4 = tf.math.reduce_mean(input_tensor=x, axis=1, keepdims=True)
assert r4.shape == (2, 1)
assert r4.dtype == tf.float32
i(r4)

# 5. tf.constant with shape kwarg
# [1, 2, 3, 4] -> shape [2, 2]
y = tf.constant(value=[1.0, 2.0, 3.0, 4.0], shape=[2, 2])
r5 = tf.reduce_mean(y)
assert r5.shape == ()
assert r5.dtype == tf.float32
j(r5)
