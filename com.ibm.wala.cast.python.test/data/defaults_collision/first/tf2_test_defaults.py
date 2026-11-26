# Probe for wala/ML#1337: this script and `second/tf2_test_defaults.py`, of the same file name,
# each define `main`, with a `None` default here and a `False` default there at the same position.
# A union of their defaults gives the batch size here the other script's `False`.
import tensorflow as tf


def consume(t):
    assert t.shape == (2, 3)
    assert t.dtype == tf.float32


def main(bs=None):
    batch_size = bs or 2
    ds = tf.data.Dataset.from_tensor_slices((tf.ones([4, 3]), tf.ones([4], tf.int32)))
    ds = ds.padded_batch(batch_size, padded_shapes=([3], []), drop_remainder=True)
    for x, _ in ds:
        consume(x)


if __name__ == "__main__":
    main()
