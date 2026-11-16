# A dataset element whose members many values hold: each call result below holds every member,
# since the helpers the members pass through are shared, so each value is read through one
# generator per member (wala/ML#1273).
import tensorflow as tf


def prepare(features, labels):
    return {
        "k0": features["k0"] * 2,
        "k1": features["k1"] * 2,
        "k2": features["k2"] * 2,
        "k3": features["k3"] * 2,
        "k4": features["k4"] * 2,
        "k5": features["k5"] * 2,
        "k6": features["k6"] * 2,
        "k7": features["k7"] * 2,
    }, labels


def inner(x):
    return x


def outer(x):
    return inner(x)


def consume(x):
    assert x.shape == (4, 3)
    assert x.dtype == tf.int32


def main():
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "k0": tf.ones([8, 3], tf.int32),
                "k1": tf.ones([8, 3], tf.int32),
                "k2": tf.ones([8, 3], tf.int32),
                "k3": tf.ones([8, 3], tf.int32),
                "k4": tf.ones([8, 3], tf.int32),
                "k5": tf.ones([8, 3], tf.int32),
                "k6": tf.ones([8, 3], tf.int32),
                "k7": tf.ones([8, 3], tf.int32),
            },
            tf.ones([8], tf.int32),
        )
    )
    dataset = dataset.map(prepare)
    dataset = dataset.shuffle(8)
    dataset = dataset.batch(4)
    dataset = dataset.prefetch(1)
    for features, labels in dataset:
        consume(outer(features["k0"]))
        consume(outer(features["k1"]))
        consume(outer(features["k2"]))
        consume(outer(features["k3"]))
        consume(outer(features["k4"]))
        consume(outer(features["k5"]))
        consume(outer(features["k6"]))
        consume(outer(features["k7"]))
        consume(outer(features["k0"]))
        consume(outer(features["k1"]))
        consume(outer(features["k2"]))
        consume(outer(features["k3"]))
        consume(outer(features["k4"]))
        consume(outer(features["k5"]))
        consume(outer(features["k6"]))
        consume(outer(features["k7"]))


main()
