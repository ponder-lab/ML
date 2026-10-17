import tensorflow as tf

# wala/ML#464: `tape.gradient(loss, sources)` returns a list of gradients for a list of sources,
# a tuple for a tuple, and a tensor for a tensor. Each result is handed to its own consumer.

w1 = tf.Variable(tf.ones([2]))
w2 = tf.Variable(tf.ones([3]))


def take_list(grads):
    assert isinstance(grads, list)
    return grads


def take_tuple(grads):
    assert isinstance(grads, tuple)
    return grads


def take_single(grad):
    assert isinstance(grad, tf.Tensor) and grad.shape == (2,)
    return grad


def scale(g):
    assert g.dtype == tf.float32
    return g * 2.0


def loss_fn():
    return tf.reduce_sum(w1) + tf.reduce_sum(w2)


with tf.GradientTape(persistent=True) as tape:
    loss = loss_fn()

as_list = tape.gradient(loss, [w1, w2])
take_list(as_list)
for g in as_list:
    scale(g)
take_tuple(tape.gradient(loss, (w1, w2)))
take_single(tape.gradient(loss, w1))
