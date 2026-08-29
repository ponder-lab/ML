# Probe for wala/ML#810: a negative `padded_shapes` entry declares "pad to the longest element in
# the batch", not an extent. Read literally it yields a negative dimension, which no shape can have
# and which a specification written from it would be rejected for.
import tensorflow as tf


def consume_padded_dynamic(a):
    pass


def consume_padded_known(b):
    pass


def rows():
    yield [1, 2, 3]
    yield [4, 5]


# The upstream extent is itself variable, so padding to the longest varies per batch.
variable = tf.data.Dataset.from_generator(
    rows, output_types=tf.int32, output_shapes=[None]
)
variable = variable.padded_batch(2, padded_shapes=[-1])

for element in variable.take(1):
    assert element.shape.rank == 2, element.shape
    assert element.dtype == tf.int32, element.dtype
    consume_padded_dynamic(element)


# The upstream extent is fixed, so padding to the longest is that extent. The runtime's own static
# shape reports None here only because it does not track the upstream size.
known = tf.data.Dataset.from_tensor_slices(tf.ones((8, 3), dtype=tf.int64))
known = known.padded_batch(4, padded_shapes=[-1])

for element in known.take(1):
    assert element.shape.rank == 2, element.shape
    assert element.shape[1] == 3, element.shape
    assert element.dtype == tf.int64, element.dtype
    consume_padded_known(element)
