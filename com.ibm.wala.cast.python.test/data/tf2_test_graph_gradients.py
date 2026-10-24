# Test https://github.com/wala/ML/issues/1144: the gradient APIs that work only while a graph is
# being built are modeled, so their calls have callees and their gradients are typed.
import tensorflow as tf

v = tf.Variable(1.0)
optimizer = tf.keras.optimizers.SGD()


def consume_get_gradients(g):
    assert g.shape == () and g.dtype == tf.float32
    return g


def consume_gradients(g):
    assert g.shape == () and g.dtype == tf.float32
    return g


def consume_backend_gradients(g):
    assert g.shape == () and g.dtype == tf.float32
    return g


@tf.function
def step():
    loss = v * v
    consume_get_gradients(optimizer.get_gradients(loss, [v])[0])
    consume_gradients(tf.gradients(loss, [v])[0])
    consume_backend_gradients(tf.keras.backend.gradients(loss, [v])[0])
    optimizer.get_updates(loss, [v])


step()
