# Test https://github.com/wala/ML/issues/1214: the element of `flat_map` and `interleave` is any
# element of the datasets their function returns, the function called on the receiver's element.
import os
import random
import tempfile

import tensorflow as tf


def consume_flat_mapped(x):
    pass


def consume_interleaved(x):
    pass


def consume_flat_mapped_pair_x(x):
    pass


def consume_flat_mapped_pair_y(y):
    pass


def consume_flat_mapped_pair(element):
    pass


def consume_unbatched(x):
    pass


def consume_either(x):
    pass


def consume_flat_mapped_batched(x):
    pass


def consume_interleaved_batched(x):
    pass


def consume_lines(line):
    pass


def consume_function_argument(v):
    pass


def consume_fit_x(x):
    pass


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))
pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)


def to_dataset(v):
    consume_function_argument(v)
    return tf.data.Dataset.from_tensors(v)


for x in flat.flat_map(to_dataset):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_flat_mapped(x)

for x in flat.interleave(lambda v: tf.data.Dataset.from_tensors(v), cycle_length=2):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_interleaved(x)

for x, y in pairs.flat_map(lambda a, b: tf.data.Dataset.from_tensors((a, b))):
    assert x.shape == (4,) and x.dtype == tf.float32
    assert y.shape == (3,) and y.dtype == tf.int32
    consume_flat_mapped_pair_x(x)
    consume_flat_mapped_pair_y(y)

for element in pairs.flat_map(lambda a, b: tf.data.Dataset.from_tensors((a, b))):
    assert isinstance(element, tuple) and len(element) == 2
    consume_flat_mapped_pair(element)

for x in flat.batch(2).flat_map(lambda b: tf.data.Dataset.from_tensor_slices(b)):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_unbatched(x)

other = tf.data.Dataset.from_tensor_slices(tf.ones([8, 5]))
for x in flat.flat_map(
    lambda v: (
        tf.data.Dataset.from_tensors(v) if random.random() > 0.5 else other.take(1)
    )
):
    assert x.shape in ((4,), (5,)) and x.dtype == tf.float32
    consume_either(x)

# The cardinality of a `flat_map` is unknown, so a batch of it may be short.
for x in flat.flat_map(lambda v: tf.data.Dataset.from_tensors(v)).batch(3):
    assert x.shape in ((3, 4), (2, 4)) and x.dtype == tf.float32
    consume_flat_mapped_batched(x)

for x in flat.interleave(
    lambda v: tf.data.Dataset.from_tensors(v), cycle_length=2
).batch(3):
    assert x.shape in ((3, 4), (2, 4)) and x.dtype == tf.float32
    consume_interleaved_batched(x)

directory = tempfile.mkdtemp()
files = []
for i in range(2):
    path = os.path.join(directory, f"lines{i}.txt")
    with open(path, "w") as f:
        f.write("a\nb\n")
    files.append(path)

for line in tf.data.Dataset.from_tensor_slices(files).interleave(
    lambda f: tf.data.TextLineDataset(f), cycle_length=2
):
    assert line.shape == () and line.dtype == tf.string
    consume_lines(line)


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, x):
        consume_fit_x(x)
        return self.dense(x)


model = Model()
model.compile(optimizer="sgd", loss="mse")
model.fit(
    pairs.flat_map(lambda a, b: tf.data.Dataset.from_tensors((a, tf.ones([1])))).batch(
        2
    ),
    epochs=1,
    verbose=0,
)


def consume_fifth(e):
    pass


def consume_not_a_dataset(x):
    pass


fives = tf.data.Dataset.from_tensor_slices(
    (
        tf.ones([8, 2]),
        tf.ones([8, 2]),
        tf.ones([8, 2]),
        tf.ones([8, 2]),
        tf.ones([8, 6]),
    )
)


def fifth_dataset(a, b, c, d, e=tf.zeros([7], tf.int32)):
    return tf.data.Dataset.from_tensors(e)


# A tuple element longer than the analysis spreads is passed whole to no call, so the element is
# unknown rather than the default's.
for e in fives.flat_map(fifth_dataset):
    assert e.shape == (6,) and e.dtype == tf.float32
    consume_fifth(e)

# A function that returns no dataset fails when the dataset is built, so it has no element.
try:
    for x in flat.flat_map(lambda v: v):
        consume_not_a_dataset(x)
    raise AssertionError("flat_map accepted a function returning a tensor")
except TypeError:
    pass
