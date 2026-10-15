# Test for https://github.com/wala/ML/issues/1075: a key read of a dict a comprehension builds over a
# dict literal.
import tensorflow as tf


def audios_sink(x):
    assert x.shape == (2, 3, 1)
    assert x.dtype == tf.float32
    return x


def sliced(d):
    copied = {k: v[0:2] for k, v in d.items()}
    audios_sink(tf.expand_dims(copied["audios"], -1))


sliced(
    {
        "audios": tf.ones((4, 3), dtype=tf.float32),
        "ids": tf.zeros((4, 5), dtype=tf.int32),
    }
)
