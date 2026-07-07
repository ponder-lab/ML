# Probe for the bare generator/next transit: a tensor yielded by a generator function, obtained
# via next() with tuple unpacking, flows directly to a downstream function with no model in
# between. Isolates the generator dataflow from the model-forward shape.
import tensorflow as tf


def consume(imgs):
    assert imgs.shape == (4, 4)
    assert imgs.dtype == tf.float32


def make_batches():
    while True:
        yield tf.ones((4, 4)), tf.zeros((4,))


batches = make_batches()
imgs, labels = next(batches)
consume(imgs)
