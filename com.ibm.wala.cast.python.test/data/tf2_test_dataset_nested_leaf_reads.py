# A dataset element's members passed through a chain of transformations whose functions return
# them, so each member is a leaf whose reading reads an upstream leaf in turn (wala/ML#1270).
import tensorflow as tf


def pair(f, l):
    return f, l


def batched_pair(f, l):
    return f, l


def indexed(i, element):
    return element


def consume(x):
    assert x.shape == (1, 2, 3)
    assert x.dtype == tf.float32


def consume_label(y):
    assert y.shape == (1, 2)
    assert y.dtype == tf.int32


def main():
    features = tf.ones([4, 3])
    labels = tf.zeros([4], dtype=tf.int32)
    dataset = (
        tf.data.Dataset.from_tensor_slices((features, labels))
        .map(pair)
        .batch(2)
        .map(batched_pair)
        .padded_batch(1)
        .enumerate()
        .map(indexed)
    )
    for x, y in dataset:
        consume(x)
        consume(tf.identity(x))
        consume_label(y)


main()
