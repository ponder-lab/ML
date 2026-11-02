# The element of a `tf.data.Dataset` built from a structure, reached by iteration and by `next`: a
# tuple or a dict of tensors, whose members are tensors and which is not one itself.
import tensorflow as tf


def pairs():
    return tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8]))
    ).batch(2)


def keyed():
    return tf.data.Dataset.from_tensor_slices(
        ({"a": tf.ones([8, 4]), "b": tf.ones([8, 3], tf.int32)}, tf.zeros([8]))
    ).batch(2)


def consume_loop_element(element):
    assert isinstance(element, tuple) and len(element) == 2


def consume_loop_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def consume_loop_x(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def consume_loop_mask(mask):
    assert mask.shape == (2, 3) and mask.dtype == tf.float32


def consume_loop_y(y):
    assert y.shape == (2,) and y.dtype == tf.float32


for element in pairs():
    consume_loop_element(element)

for inputs, y in pairs():
    consume_loop_inputs(inputs)
    x, mask = inputs
    consume_loop_x(x)
    consume_loop_mask(mask)
    consume_loop_y(y)


def consume_next_element(element):
    assert isinstance(element, tuple) and len(element) == 2


consume_next_element(next(iter(pairs())))


def consume_keyed_features(features):
    assert isinstance(features, dict) and sorted(features) == ["a", "b"]


def consume_keyed_a(a):
    assert a.shape == (2, 4) and a.dtype == tf.float32


def consume_keyed_b(b):
    assert b.shape == (2, 3) and b.dtype == tf.int32


for features, y in keyed():
    consume_keyed_features(features)
    consume_keyed_a(features["a"])
    consume_keyed_b(features["b"])


def consume_flat_keyed_a(a):
    assert a.shape == (2, 4) and a.dtype == tf.float32


flat_keyed = tf.data.Dataset.from_tensor_slices({"a": tf.ones([8, 4])}).batch(2)
consume_flat_keyed_a(next(iter(flat_keyed))["a"])


# A list is no structure: `from_tensor_slices` stacks it into one tensor.
def consume_stacked(stacked):
    assert stacked.shape == (2, 4) and stacked.dtype == tf.float32


for stacked in tf.data.Dataset.from_tensor_slices([tf.ones([4]), tf.ones([4])]).batch(
    2
):
    consume_stacked(stacked)


# Transforms that leave the element unchanged keep its structure.
def consume_shuffled_mask(mask):
    assert mask.shape == (2, 3) and mask.dtype == tf.float32


for (x, mask), y in pairs().shuffle(4).repeat(2).take(3).prefetch(1):
    consume_shuffled_mask(mask)


# An element mapped by user code, then batched.
def consume_mapped_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def consume_mapped_b(b):
    assert b.shape == (2, 3) and b.dtype == tf.float32


mapped = (
    tf.data.Dataset.from_tensor_slices(
        (tf.ones([8, 4]), tf.ones([8, 3]), tf.zeros([8]))
    )
    .map(lambda a, b, y: ((a, b), y))
    .batch(2)
)
for inputs, y in mapped:
    consume_mapped_inputs(inputs)
    a, b = inputs
    consume_mapped_b(b)
