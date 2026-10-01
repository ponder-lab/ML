# Test for a shape argument whose value holds itself: a list rebuilt in a loop around its previous
# value is one abstract object that contains itself, so reading it as a shape must not recurse
# forever.
import tensorflow as tf


def consume(x):
    pass


shape = [2]
for _ in range(2):
    shape = [shape, 2]
try:
    consume(tf.ones(shape))
except (TypeError, ValueError):
    pass


def consume_spec(x):
    pass


# The same cycle through a spec's shape field: the list arm reads the spec, whose shape is the list.
spec = tf.TensorSpec([2])
try:
    for _ in range(2):
        spec = tf.TensorSpec([spec, 2])
    consume_spec(tf.ones(spec.shape))
except (TypeError, ValueError):
    pass


def consume_dict(x):
    pass


# The same cycle through a dict-structured shape: each value of `padded_shapes` is read as a shape.
shapes = {"h": [None]}
for _ in range(2):
    shapes = {"h": shapes}
try:
    for element in tf.data.Dataset.from_tensors({"h": tf.ones(3)}).padded_batch(
        2, padded_shapes=shapes
    ):
        consume_dict(element["h"])
except (TypeError, ValueError):
    pass
