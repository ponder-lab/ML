import tensorflow as tf


def func2(t):
    pass


@tf.function
def func():
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    assert a.shape == (2, 2)
    assert a.dtype == tf.float32

    z = tf.zeros_like(a)
    assert z.shape == (2, 2)
    assert z.dtype == tf.float32

    func2(z)


if __name__ == "__main__":
    func()
