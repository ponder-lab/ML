# Test https://github.com/wala/ML/issues/1190: the element of `Dataset.enumerate()` is a tuple of an
# int64 scalar index and the receiver's element, whose members keep their own types.
import random

import tensorflow as tf


def consume_flat_index(i):
    pass


def consume_flat_x(x):
    pass


def consume_pair_index(i):
    pass


def consume_pair_x(x):
    pass


def consume_pair_y(y):
    pass


def consume_pair_element(e):
    pass


def consume_batched_index(i):
    pass


def consume_batched_x(x):
    pass


def consume_batched_y(y):
    pass


def consume_dict_index(i):
    pass


def consume_dict_a(a):
    pass


def consume_dict_b(b):
    pass


def pair_dataset():
    return tf.data.Dataset.from_tensor_slices(
        (tf.ones([6, 4]), tf.zeros([6, 2], dtype=tf.int32))
    )


def flat():
    for i, x in tf.data.Dataset.from_tensor_slices(tf.ones([5, 3])).enumerate():
        assert i.shape == () and i.dtype == tf.int64
        assert x.shape == (3,) and x.dtype == tf.float32
        consume_flat_index(i)
        consume_flat_x(x)


def pair():
    for i, (x, y) in pair_dataset().enumerate():
        assert i.shape == () and i.dtype == tf.int64
        assert x.shape == (4,) and x.dtype == tf.float32
        assert y.shape == (2,) and y.dtype == tf.int32
        consume_pair_index(i)
        consume_pair_x(x)
        consume_pair_y(y)


def pair_element():
    for i, e in pair_dataset().enumerate():
        assert isinstance(e, tuple) and len(e) == 2
        consume_pair_element(e)


def batched():
    for i, (x, y) in pair_dataset().enumerate().batch(2):
        assert i.shape == (2,) and i.dtype == tf.int64
        assert x.shape == (2, 4) and x.dtype == tf.float32
        assert y.shape == (2, 2) and y.dtype == tf.int32
        consume_batched_index(i)
        consume_batched_x(x)
        consume_batched_y(y)


def dict_element():
    dataset = tf.data.Dataset.from_tensor_slices(
        {"a": tf.ones([5, 3]), "b": tf.zeros([5], dtype=tf.int32)}
    )
    for i, d in dataset.enumerate():
        assert i.shape == () and i.dtype == tf.int64
        a = d["a"]
        assert a.shape == (3,) and a.dtype == tf.float32
        b = d["b"]
        assert b.shape == () and b.dtype == tf.int32
        consume_dict_index(i)
        consume_dict_a(a)
        consume_dict_b(b)


flat()
pair()
pair_element()
batched()
dict_element()


def consume_generated_x(x):
    pass


def gen_pairs():
    for _ in range(4):
        yield tf.ones([3]), tf.zeros([2], tf.int32)


generated = tf.data.Dataset.from_generator(
    gen_pairs,
    output_signature=(
        tf.TensorSpec(shape=(3,), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
    ),
)
for i, (x, y) in generated.enumerate():
    assert x.shape == (3,) and x.dtype == tf.float32
    consume_generated_x(x)


def consume_zipped_index(i):
    pass


def consume_zipped_x(x):
    pass


def consume_zipped_y(y):
    pass


def consume_zipped_inner_index(i):
    pass


def consume_zipped_inner_x(x):
    pass


def consume_zipped_inner_y(y):
    pass


def consume_mixed_index(x):
    pass


def consume_mixed_x(x):
    pass


def consume_mixed_member(x):
    pass


def consume_unmodeled_index(x):
    pass


def consume_unmodeled_x(x):
    pass


def consume_shuffled_mixed_x(x):
    pass


def consume_enumerated_map_x(x):
    pass


def consume_enumerated_mixed_map_x(x):
    pass


floats = tf.data.Dataset.from_tensor_slices(tf.ones([6, 4]))
ints = tf.data.Dataset.from_tensor_slices(tf.zeros([6, 2], tf.int32))
for i, (x, y) in tf.data.Dataset.zip((floats, ints)).enumerate():
    assert i.shape == () and i.dtype == tf.int64
    assert x.shape == (4,) and x.dtype == tf.float32
    assert y.shape == (2,) and y.dtype == tf.int32
    consume_zipped_index(i)
    consume_zipped_x(x)
    consume_zipped_y(y)
for (i, x), y in tf.data.Dataset.zip((floats.enumerate(), ints)):
    assert i.shape == () and i.dtype == tf.int64
    assert x.shape == (4,) and x.dtype == tf.float32
    assert y.shape == (2,) and y.dtype == tf.int32
    consume_zipped_inner_index(i)
    consume_zipped_inner_x(x)
    consume_zipped_inner_y(y)

# A receiver that may be either of two datasets: the element is the join of both
# (https://github.com/wala/ML/issues/1204).
picked = random.random() > 0.5
threes = tf.data.Dataset.from_tensor_slices(tf.zeros([6, 3], tf.int32))
mixed = floats if picked else threes
for i, x in mixed.enumerate():
    assert i.shape == () and i.dtype == tf.int64
    assert x.shape in ((4,), (3,))
    consume_mixed_index(i)
    consume_mixed_x(x)
for e in mixed.enumerate():
    consume_mixed_member(e[1])


def fives():
    for _ in range(3):
        yield tf.ones((5,))


# One of the two datasets declares no structure, so its element is not mirrored.
unmodeled = (
    floats if picked else tf.data.Dataset.from_generator(fives, output_types=tf.float32)
)
for i, x in unmodeled.enumerate():
    assert i.shape == () and i.dtype == tf.int64
    assert x.shape in ((4,), (5,)) and x.dtype == tf.float32
    consume_unmodeled_index(i)
    consume_unmodeled_x(x)

# A pass-through transformation of the receiver that may be either dataset.
for i, x in mixed.shuffle(2).enumerate():
    assert x.shape in ((4,), (3,))
    consume_shuffled_mixed_x(x)

# A mapped dataset is enumerated through its mapped element, alone or as one of two receivers.
for i, x in threes.map(lambda v: v + 1).enumerate():
    assert x.shape == (3,) and x.dtype == tf.int32
    consume_enumerated_map_x(x)
for i, x in (floats if picked else threes.map(lambda v: v + 1)).enumerate():
    assert x.shape in ((4,), (3,))
    consume_enumerated_mixed_map_x(x)
