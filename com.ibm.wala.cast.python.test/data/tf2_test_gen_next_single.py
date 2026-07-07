# Narrowing probe for the generator/next transit: the generator yields a SINGLE tensor (no tuple),
# retrieved via next() with no unpacking. Discriminates the generator-yield dataflow itself from
# the tuple-unpack of a yielded pair.
import tensorflow as tf


def consume(imgs):
    assert imgs.shape == (4, 4)
    assert imgs.dtype == tf.float32


def make_batches():
    while True:
        yield tf.ones((4, 4))


batches = make_batches()
imgs = next(batches)
consume(imgs)
