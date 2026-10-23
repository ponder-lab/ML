import tensorflow as tf


def consume_float_list(t):
    pass


def consume_float_scalar(t):
    pass


def consume_int_scalar(t):
    pass


def consume_bool_list(t):
    pass


def consume_number(t):
    pass


n = 6
s = "3"

# A Python float built by `float(...)` converts to float32.
a = tf.constant([float(n)])
assert a.shape == (1,)
assert a.dtype == tf.float32
consume_float_list(a)

b = tf.constant(float(n))
assert b.shape == ()
assert b.dtype == tf.float32
consume_float_scalar(b)

# A Python int built by `int(...)` converts to int32.
c = tf.constant(int(s))
assert c.shape == ()
assert c.dtype == tf.int32
consume_int_scalar(c)

# A Python bool built by `bool(...)` converts to bool.
d = tf.constant([bool(n)])
assert d.shape == (1,)
assert d.dtype == tf.bool
consume_bool_list(d)


# The builtins' results are Python numbers, not tensors.
x = float(n)
assert not tf.is_tensor(x)
consume_number(x)
consume_number(int(s) - 1)
consume_number(bool(n))
