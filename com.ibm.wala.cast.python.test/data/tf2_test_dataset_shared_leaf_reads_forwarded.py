# As tf2_test_dataset_shared_leaf_reads.py, with both maps calling the shared function through one
# forwarding function, so the shared function's context is the same at both maps and their
# readings still form a cycle (wala/ML#1270, wala/ML#1305).
import tensorflow as tf


def pair(f, l):
    return f, l


def forward(f, l):
    return pair(f, l)


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
        .map(forward)
        .batch(2)
        .map(forward)
        .padded_batch(1)
        .enumerate()
        .map(indexed)
    )
    for x, y in dataset:
        consume(x)
        consume(tf.identity(x))
        consume_label(y)


main()
