import tensorflow as tf


def func2(t):
    pass


@tf.function
def func():
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    assert a.shape == (2, 2)

    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    assert b.shape == (2, 2)

    c = tf.matmul(a, b)
    tensor = tf.experimental.numpy.ndarray(c.op, 0, tf.float32)
    func2(tensor)


func()
