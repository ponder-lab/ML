import tensorflow as tf


def f(a):
    pass


# Regression fixture for wala/ML#464: when `sources` is a list (the common
# Keras pattern `tape.gradient(loss, model.trainable_variables)`), the runtime
# returns a parallel list of fresh tensors—one per source. The analyzer should
# resolve `grads[i]` to the shape/dtype of the i-th source.
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

# Sink: pass the first gradient to `f`. The analyzer should classify `f`'s
# parameter as a tensor with shape (2,) and dtype float32, inherited from the
# first source `w1` (which itself is a [2]-shaped float32 tensor).
f(grads[0])
