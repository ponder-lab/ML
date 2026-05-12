import tensorflow as tf


def f(a):
    pass


# Companion to `tf2_test_gradient_list.py`: passes BOTH gradients to `f` in
# separate calls, so the analyzer's tensor-type for `f`'s parameter must be the
# union of the per-source types. With the `Gradient` `TupleElementProvider`
# implementation, both `grads[0]` (shape (2,) from `w1`) and `grads[1]`
# (shape (1, 1) from `w2`) should appear in the parameter's type set.
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

f(grads[0])
f(grads[1])
