import tensorflow as tf


def f(a):
    pass


# Regression fixture for wala/ML#464: when `sources` is a list (the common
# Keras pattern `tape.gradient(loss, model.trainable_variables)`), the runtime
# returns a parallel list of fresh tensors—one per source. The analyzer should
# resolve `grads[i]` to the shape/dtype of the i-th source. Passing both
# gradients to `f` across two call sites exercises both per-index resolutions;
# `f`'s parameter type is the union of the two source types.
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
