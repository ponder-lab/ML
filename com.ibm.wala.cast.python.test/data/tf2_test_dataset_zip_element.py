# Test https://github.com/wala/ML/issues/164: `zip` yields the structure it is given with each
# dataset replaced by that dataset's element, so each member keeps its own type.
import tensorflow as tf


def consume_pair(p):
    pass


def consume_a(a):
    pass


def consume_b(b):
    pass


def consume_batched_a(a):
    pass


def consume_batched_b(b):
    pass


def consume_nested_x(x):
    pass


def consume_nested_mask(mask):
    pass


def consume_keyed_a(a):
    pass


def consume_mapped_a(a):
    pass


def consume_mapped_b(b):
    pass


def consume_keyed_b(b):
    pass


def consume_sized_uneven_a(a):
    pass


def consume_sized_keyed_a(a):
    pass


def consume_sized_unknown_a(a):
    pass


def sized_rows():
    for _ in range(8):
        yield tf.zeros((3,), tf.int32)


floats = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))
ints = tf.data.Dataset.from_tensor_slices(tf.zeros([8, 3], tf.int32))

for p in tf.data.Dataset.zip((floats, ints)):
    assert isinstance(p, tuple) and len(p) == 2
    consume_pair(p)

for a, b in tf.data.Dataset.zip((floats, ints)):
    assert a.shape == (4,) and a.dtype == tf.float32
    consume_a(a)
    assert b.shape == (3,) and b.dtype == tf.int32
    consume_b(b)

for a, b in tf.data.Dataset.zip((floats, ints)).batch(2):
    assert a.shape == (2, 4) and a.dtype == tf.float32
    consume_batched_a(a)
    assert b.shape == (2, 3) and b.dtype == tf.int32
    consume_batched_b(b)

pairs = tf.data.Dataset.from_tensor_slices((tf.ones([8, 5]), tf.ones([8, 2])))
for (x, mask), b in tf.data.Dataset.zip((pairs, ints)):
    assert x.shape == (5,) and x.dtype == tf.float32
    consume_nested_x(x)
    assert mask.shape == (2,) and mask.dtype == tf.float32
    consume_nested_mask(mask)

for e in tf.data.Dataset.zip({"a": floats, "b": ints}):
    a = e["a"]
    assert a.shape == (4,) and a.dtype == tf.float32
    consume_keyed_a(a)
    b = e["b"]
    assert b.shape == (3,) and b.dtype == tf.int32
    consume_keyed_b(b)

# A zipped dataset whose element is not copied keeps the element its generator types.
for a, b in tf.data.Dataset.zip((floats.map(lambda v: v * 2), ints)):
    assert a.shape == (4,) and a.dtype == tf.float32
    consume_mapped_a(a)
    assert b.shape == (3,) and b.dtype == tf.int32
    consume_mapped_b(b)

# A zip has as many elements as its shortest dataset
# (https://github.com/wala/ML/issues/1203): 5 elements batch by 2 into two full batches and one of 1.
shorter = tf.data.Dataset.from_tensor_slices(tf.zeros([5, 3], tf.int32))
for a, b in tf.data.Dataset.zip((floats, shorter)).batch(2):
    assert a.shape in ((2, 4), (1, 4)) and a.dtype == tf.float32
    consume_sized_uneven_a(a)
for e in tf.data.Dataset.zip({"a": floats, "b": ints}).batch(2):
    assert e["a"].shape == (2, 4) and e["a"].dtype == tf.float32
    consume_sized_keyed_a(e["a"])
# A dataset of no known size leaves the zip's size unknown.
unsized = tf.data.Dataset.from_generator(
    sized_rows, output_signature=tf.TensorSpec(shape=(3,), dtype=tf.int32)
)
for a, b in tf.data.Dataset.zip((floats, unsized)).batch(3):
    assert a.shape in ((3, 4), (2, 4)) and a.dtype == tf.float32
    consume_sized_unknown_a(a)
