import tensorflow as tf


def func2(t):
    pass


@tf.function
def func():
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    assert a.shape == (2, 2)
    assert a.dtype == tf.float32

    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    assert b.shape == (2, 2)
    assert b.dtype == tf.float32

    c = tf.matmul(a, b)
    assert c.shape == (2, 2)
    assert c.dtype == tf.float32

    tensor = tf.experimental.numpy.ndarray(c.op, 0, tf.float32)
    assert tensor.shape == (2, 2)
    assert tensor.dtype == tf.float32

    func2(tensor)


func()
