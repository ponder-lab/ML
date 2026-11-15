# A dataset element's members, relayed through several functions, so many values hold the same
# mirrored leaves, which the analysis reads once (wala/ML#1269).
import tensorflow as tf


def relay1(x):
    return relay2(x)


def relay2(x):
    return relay3(x)


def relay3(x):
    return x


def relay_label(y):
    return y


def consume(x):
    assert x.shape == (2, 3)
    assert x.dtype == tf.float32


def consume_label(y):
    assert y.shape == (2,)
    assert y.dtype == tf.int32


def main():
    features = tf.ones([4, 3])
    labels = tf.zeros([4], dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(2)
    for x, y in dataset:
        a = relay1(x)
        b = relay_label(y)
        consume(a + 1.0)
        consume(tf.identity(a))
        consume_label(b)


main()
