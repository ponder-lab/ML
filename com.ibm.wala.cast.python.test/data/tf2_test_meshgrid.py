import tensorflow as tf


def f(a):
    pass


# `tf.meshgrid(*xi)` returns N tensors (one per input). All output
# tensors share the broadcast of input shapes and the first input's
# dtype. The fixture exercises destructuring access into the tuple
# result so the `TupleElementProvider` wrap fires on each element.
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([10.0, 20.0])
X, Y = tf.meshgrid(x, y)
assert isinstance(X, tf.Tensor)
assert isinstance(Y, tf.Tensor)
assert X.dtype == tf.float32
assert Y.dtype == tf.float32
f(X)
f(Y)
