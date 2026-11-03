# The element of a `tf.data.Dataset` built from a structure, reached by iteration and by `next`: a
# tuple or a dict of tensors, whose members are tensors and which is not one itself.
import sys

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


# A value that is a dataset's element on one arm and an ordinary tensor on the other.
def consume_either_cast(x):
    assert x.shape in ((2, 4), (3, 5)) and x.dtype == tf.int32


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4])).batch(2)
either = next(iter(flat)) if len(sys.argv) > 5 else tf.ones([3, 5])
consume_either_cast(tf.cast(either, tf.int32))


# A member nested nine tuples deep, past the depth the element's copy records.
def consume_deep(x):
    assert x.shape == (4,) and x.dtype == tf.float32


deep = tf.data.Dataset.from_tensor_slices((((((((((tf.ones([8, 4]),),),),),),),),),))
for d in deep:
    consume_deep(d[0][0][0][0][0][0][0][0][0])


# A member read off an element, then used: cast, then arithmetic.
def consume_member_cast_scaled(x):
    pass


def consume_keyed_member_cast_scaled(b):
    pass


def consume_next_member_shifted(a):
    pass


for (x, mask), y in pairs():
    x_cast_scaled = tf.cast(x, tf.int32) * 2
    assert x_cast_scaled.shape == (2, 4) and x_cast_scaled.dtype == tf.int32
    consume_member_cast_scaled(x_cast_scaled)

for features, y in keyed():
    b_cast_scaled = tf.cast(features["b"], tf.float32) / 2.0
    assert b_cast_scaled.shape == (2, 3) and b_cast_scaled.dtype == tf.float32
    consume_keyed_member_cast_scaled(b_cast_scaled)

a_shifted = next(iter(flat_keyed))["a"] + 1.0
assert a_shifted.shape == (2, 4) and a_shifted.dtype == tf.float32
consume_next_member_shifted(a_shifted)


# A dict key of digits too long for an int.
def consume_long_digit_key(x):
    pass


long_keyed = tf.data.Dataset.from_tensor_slices(
    {"12345678901234567890": tf.ones((4, 3), dtype=tf.float32)}
)
for long_element in long_keyed:
    long_member = long_element["12345678901234567890"]
    assert long_member.shape == (3,) and long_member.dtype == tf.float32
    consume_long_digit_key(long_member)


def consume_digit_key(x):
    pass


digit_keyed = tf.data.Dataset.from_tensor_slices(
    {"0": tf.ones((4, 3), dtype=tf.float32), "b": tf.zeros((4, 2), dtype=tf.int32)}
)
for digit_element in digit_keyed:
    digit_member = digit_element["0"]
    assert digit_member.shape == (3,) and digit_member.dtype == tf.float32
    consume_digit_key(digit_member)


def consume_tuple_position(x):
    pass


positioned = tf.data.Dataset.from_tensor_slices(
    (tf.ones((4, 3), dtype=tf.float32), tf.zeros((4, 2), dtype=tf.int32))
)
for positioned_element in positioned:
    position_member = positioned_element[0]
    assert position_member.shape == (3,) and position_member.dtype == tf.float32
    consume_tuple_position(position_member)
