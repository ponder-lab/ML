# Reproducer for wala/ML#698: element flow through an explicit `iter()` call. `iter()` was modeled
# as a fresh, empty `iterator` allocation, so `next(iter(gen()))` read the generator content field
# off the fresh iterator instead of the generator object and the yielded tensor's type was dropped.
# With `iter` modeled as a pass-through of its argument, the shape/dtype transits.
import tensorflow as tf


def consume(imgs):
    assert imgs.shape == (4, 4)
    assert imgs.dtype == tf.float32


def make_batches():
    while True:
        yield tf.ones((4, 4))


batches = iter(make_batches())
imgs = next(batches)
consume(imgs)
