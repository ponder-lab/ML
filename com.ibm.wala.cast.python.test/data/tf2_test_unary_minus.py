import tensorflow as tf


def consume_int(t):
    pass


def consume_float(t):
    pass


def consume_where(t):
    pass


def consume_invert(t):
    pass


def consume_number(t):
    pass


n = 6

# Unary minus on an int32 tensor: same shape and dtype as the operand.
labels = -tf.ones(n, dtype=tf.int32)
assert labels.shape == (6,)
assert labels.dtype == tf.int32
consume_int(labels)

# Unary minus on a float tensor held in a variable.
x = tf.zeros([2, 3])
y = -x
assert y.shape == (2, 3)
assert y.dtype == tf.float32
consume_float(y)

# A negated tensor feeding `tf.where`.
cond = tf.constant([True, False, True, False, True, False])
w = tf.where(cond, tf.zeros(n, dtype=tf.int32), -tf.ones(n, dtype=tf.int32))
assert w.shape == (6,)
assert w.dtype == tf.int32
consume_where(w)

# Bitwise invert on an int32 tensor.
b = ~tf.ones(n, dtype=tf.int32)
assert b.shape == (6,)
assert b.dtype == tf.int32
consume_invert(b)

# Unary minus on Python numbers is not a tensor.
k = -3
m = -n
f = -2.5
assert not tf.is_tensor(k) and not tf.is_tensor(m) and not tf.is_tensor(f)
consume_number(k)
consume_number(m)
consume_number(f)
