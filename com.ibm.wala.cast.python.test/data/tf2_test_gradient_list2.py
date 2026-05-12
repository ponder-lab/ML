import tensorflow as tf


def f(a, b):
    pass


# Regression fixture for wala/ML#464, tighter variant of `tf2_test_gradient_list.py`:
# pass both gradients in a single call so the analyzer must resolve each
# argument's tensor type independently per its source index, rather than as a
# union across two call sites. With the `Gradient` `TupleElementProvider`
# implementation, `f`'s first parameter resolves to the type of `w1`'s gradient
# (shape (2,), float32) and the second to `w2`'s (shape (1, 1), float32).
w1 = tf.Variable(tf.constant([1.0, 2.0]))
w2 = tf.Variable(tf.constant([[3.0]]))

with tf.GradientTape() as tape:
    loss = tf.reduce_sum(w1) + tf.reduce_sum(w2)

grads = tape.gradient(loss, [w1, w2])

assert isinstance(grads, list)
assert grads[0].shape == (2,)
assert grads[0].dtype == tf.float32
assert grads[1].shape == (1, 1)
assert grads[1].dtype == tf.float32

f(grads[0], grads[1])
