# Companion probe for the generator transit: the same yielded pair consumed by for-loop
# destructuring over the generator instead of next(). Mirrors the common training-loop shape.
import tensorflow as tf


def consume(imgs):
    assert imgs.shape == (4, 4)
    assert imgs.dtype == tf.float32


def make_batches():
    for _ in range(2):
        yield tf.ones((4, 4)), tf.zeros((4,))


for imgs, labels in make_batches():
    consume(imgs)
