# Test https://github.com/wala/ML/issues/164: the element of `from_generator` has the structure
# it declares, so a tuple or dict element is no tensor and each member keeps its own type.
import tensorflow as tf


def consume_pair(p):
    pass


def consume_pair_x(x):
    pass


def consume_pair_y(y):
    pass


def consume_keyed_x(x):
    pass


def consume_keyed_y(y):
    pass


def consume_nested_x(x):
    pass


def consume_nested_mask(mask):
    pass


def consume_nested_y(y):
    pass


def consume_typed_x(x):
    pass


def consume_typed_y(y):
    pass


def consume_same_x(x):
    pass


def consume_same_y(y):
    pass


def consume_batched_x(x):
    pass


def consume_batched_y(y):
    pass


def consume_undeclared_x(x):
    pass


def pairs():
    for _ in range(4):
        yield tf.ones([4]), tf.zeros([2, 3], tf.int32)


def same_dtype():
    for _ in range(4):
        yield tf.ones([4]), tf.ones([2, 3])


def keyed():
    for _ in range(4):
        yield {"x": tf.ones([5]), "y": tf.zeros([], tf.int64)}


def nested():
    for _ in range(4):
        yield (tf.ones([6]), tf.ones([2])), tf.zeros([], tf.int32)


pair_signature = (
    tf.TensorSpec(shape=(4,), dtype=tf.float32),
    tf.TensorSpec(shape=(2, 3), dtype=tf.int32),
)

for p in tf.data.Dataset.from_generator(pairs, output_signature=pair_signature):
    assert isinstance(p, tuple) and len(p) == 2
    consume_pair(p)

for x, y in tf.data.Dataset.from_generator(pairs, output_signature=pair_signature):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_pair_x(x)
    assert y.shape == (2, 3) and y.dtype == tf.int32
    consume_pair_y(y)

keyed_signature = {
    "x": tf.TensorSpec(shape=(5,), dtype=tf.float32),
    "y": tf.TensorSpec(shape=(), dtype=tf.int64),
}

for e in tf.data.Dataset.from_generator(keyed, output_signature=keyed_signature):
    assert isinstance(e, dict)
    x = e["x"]
    assert x.shape == (5,) and x.dtype == tf.float32
    consume_keyed_x(x)
    y = e["y"]
    assert y.shape == () and y.dtype == tf.int64
    consume_keyed_y(y)

nested_signature = (
    (
        tf.TensorSpec(shape=(6,), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(), dtype=tf.int32),
)

for (x, mask), y in tf.data.Dataset.from_generator(
    nested, output_signature=nested_signature
):
    assert x.shape == (6,) and x.dtype == tf.float32
    consume_nested_x(x)
    assert mask.shape == (2,) and mask.dtype == tf.float32
    consume_nested_mask(mask)
    assert y.shape == () and y.dtype == tf.int32
    consume_nested_y(y)

for x, y in tf.data.Dataset.from_generator(
    pairs, output_types=(tf.float32, tf.int32), output_shapes=((4,), (2, 3))
):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_typed_x(x)
    assert y.shape == (2, 3) and y.dtype == tf.int32
    consume_typed_y(y)

# Both members share one dtype object, and keep their own shapes.
for x, y in tf.data.Dataset.from_generator(
    same_dtype, output_types=(tf.float32, tf.float32), output_shapes=((4,), (2, 3))
):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_same_x(x)
    assert y.shape == (2, 3) and y.dtype == tf.float32
    consume_same_y(y)

for x, y in tf.data.Dataset.from_generator(
    pairs, output_signature=pair_signature
).batch(2):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    consume_batched_x(x)
    assert y.shape == (2, 2, 3) and y.dtype == tf.int32
    consume_batched_y(y)

# A dataset declaring no structure is rejected at run time; its element is the generic one.
try:
    for x, y in tf.data.Dataset.from_generator(pairs):
        consume_undeclared_x(x)
    raise AssertionError("A dataset declaring no structure is rejected.")
except TypeError:
    pass
