# Test https://github.com/wala/ML/issues/164: the element of a dataset `tfds.load` returns under
# `as_supervised=True` is an `(input, label)` pair, so it is no tensor and each member is an object
# of its own. Without `as_supervised`, or with it false or not a literal, an element is a dict of
# the named dataset's features. TensorFlow Datasets is not installed for the test interpreter, and
# loading a dataset downloads it, so this file is not run. Under TensorFlow Datasets 4.9, an
# element of `mnist` is an image of shape `(28, 28, 1)` and dtype `uint8` and a label of shape `()`
# and dtype `int64`; without `as_supervised`, it is a dict with the keys "image" and "label".
import sys

import tensorflow as tf
import tensorflow_datasets as tfds


def consume_pair(p):
    pass


def consume_pair_x(x):
    pass


def consume_pair_y(y):
    pass


def consume_batched_x(x):
    pass


def consume_batched_y(y):
    pass


def consume_omitted(e):
    pass


def consume_false(e):
    pass


def consume_conditional(e):
    pass


def consume_zipped_pair(p):
    pass


def consume_enumerated_index(i):
    pass


def consume_enumerated_x(x):
    pass


def consume_zipped_x(x):
    pass


def consume_zipped_z(z):
    pass


supervised = tfds.load("mnist", split="train", as_supervised=True)

for p in supervised:
    assert isinstance(p, tuple) and len(p) == 2
    consume_pair(p)

for x, y in tfds.load("mnist", split="train", as_supervised=True):
    assert x.shape == (28, 28, 1) and x.dtype == tf.uint8
    consume_pair_x(x)
    assert y.shape == () and y.dtype == tf.int64
    consume_pair_y(y)

for x, y in tfds.load("mnist", split="train", as_supervised=True).batch(2):
    assert x.shape[1:] == (28, 28, 1) and x.dtype == tf.uint8
    consume_batched_x(x)
    assert y.shape[1:] == () and y.dtype == tf.int64
    consume_batched_y(y)

# Without `as_supervised`, an element is a dict of features.
for e in tfds.load("mnist", split="train"):
    assert isinstance(e, dict)
    assert e["image"].shape == (28, 28, 1) and e["image"].dtype == tf.uint8
    consume_omitted(e)

for e in tfds.load("mnist", split="train", as_supervised=False):
    assert isinstance(e, dict)
    assert e["image"].shape == (28, 28, 1) and e["image"].dtype == tf.uint8
    consume_false(e)

# An `as_supervised` that is no literal, here false at run time, reads as a dict too.
for e in tfds.load("mnist", split="train", as_supervised=len(sys.argv) > 1000):
    assert isinstance(e, dict)
    assert e["image"].shape == (28, 28, 1) and e["image"].dtype == tf.uint8
    consume_conditional(e)

# A zip of the supervised dataset with another keeps the pair as the zip's first member.
zipped = tf.data.Dataset.zip(
    (
        tfds.load("mnist", split="train", as_supervised=True),
        tf.data.Dataset.from_tensor_slices(tf.ones([60000, 4])),
    )
)
for p, z in zipped:
    assert isinstance(p, tuple) and len(p) == 2
    consume_zipped_pair(p)
    x, y = p
    assert x.shape == (28, 28, 1) and x.dtype == tf.uint8
    consume_zipped_x(x)
    assert z.shape == (4,) and z.dtype == tf.float32
    consume_zipped_z(z)

# An enumeration of the supervised dataset pairs an index with the pair.
for i, (x, y) in tfds.load("mnist", split="train", as_supervised=True).enumerate():
    assert i.shape == () and i.dtype == tf.int64
    consume_enumerated_index(i)
    assert x.shape == (28, 28, 1) and x.dtype == tf.uint8
    consume_enumerated_x(x)
