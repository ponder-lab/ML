# Test https://github.com/wala/ML/issues/1201: `window` yields datasets whose own iteration yields
# the source elements, and for a tuple element a tuple of datasets, one per component.
import os
import random
import tempfile

import numpy as np
import tensorflow as tf


def consume_window(w):
    pass


def consume_row(e):
    pass


def consume_first_window(a):
    pass


def consume_first_row(x):
    pass


def consume_second_row(y):
    pass


def consume_window_batch(b):
    pass


def consume_lines(lines):
    pass


def consume_fifth_rows(e):
    pass


def consume_sixteenth_row(r):
    pass


def consume_seventeenth_window(w):
    pass


for w in tf.data.Dataset.from_tensor_slices(tf.ones([8, 4])).window(2):
    assert isinstance(w, tf.data.Dataset)
    consume_window(w)
    for e in w:
        assert e.shape == (4,) and e.dtype == tf.float32
        consume_row(e)

# A window is a dataset, so it has the dataset methods: a batch of a window of two is one batch of
# its two elements.
for w in tf.data.Dataset.from_tensor_slices(tf.ones([8, 4])).window(2):
    for b in w.batch(2):
        assert b.shape == (2, 4) and b.dtype == tf.float32
        consume_window_batch(b)

pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)
for a, b in pairs.window(2):
    assert isinstance(a, tf.data.Dataset) and isinstance(b, tf.data.Dataset)
    consume_first_window(a)
    for x in a:
        assert x.shape == (4,) and x.dtype == tf.float32
        consume_first_row(x)
    for y in b:
        assert y.shape == (3,) and y.dtype == tf.int32
        consume_second_row(y)

# The element of a `TextLineDataset` is not mirrored, so a window of a batch of its lines reads its
# rows, the batches, through the window's own generator.
directory = tempfile.mkdtemp()
path = os.path.join(directory, "lines.txt")
with open(path, "w") as f:
    f.write("a\nb\nc\n")
for w in tf.data.TextLineDataset(path).batch(2).window(2):
    for lines in w:
        assert lines.shape in ((2,), (1,)) and lines.dtype == tf.string
        consume_lines(lines)


# A window of the fifth component of a five-component element: its rows, turned into arrays and
# concatenated, keep the declared dtype (wala/ML#1221).
def five():
    for _ in range(6):
        n = random.randint(1, 4)
        yield (
            np.arange(n, dtype=np.int32),
            np.int32(0),
            np.zeros([2, n], np.int32),
            np.ones([n], np.int32),
            np.zeros([n], np.int32),
        )


def merge(window):
    rows = [np.array(item) for item in window]
    return tf.concat(rows, 0)


generated = tf.data.Dataset.from_generator(
    five, (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
)
for a, b, c, d, e in generated.shuffle(10).prefetch(2).window(2):
    merged = merge(e)
    assert merged.dtype == tf.int32 and merged.shape.rank == 1
    consume_fifth_rows(merged)

# A seventeen-component element: the window of position 15 is the last the analysis makes, and
# position 16 has none (wala/ML#1221).
wide = tf.data.Dataset.from_tensor_slices(
    (
        tf.ones([8, 1]),
        tf.ones([8, 2]),
        tf.ones([8, 3]),
        tf.ones([8, 4]),
        tf.ones([8, 5]),
        tf.ones([8, 6]),
        tf.ones([8, 7]),
        tf.ones([8, 8]),
        tf.ones([8, 9]),
        tf.ones([8, 10]),
        tf.ones([8, 11]),
        tf.ones([8, 12]),
        tf.ones([8, 13]),
        tf.ones([8, 14]),
        tf.ones([8, 15]),
        tf.ones([8, 16]),
        tf.ones([8, 17]),
    )
)
for parts in wide.window(2):
    for r in parts[15]:
        assert r.shape == (16,) and r.dtype == tf.float32
        consume_sixteenth_row(r)
    assert isinstance(parts[16], tf.data.Dataset)
    consume_seventeenth_window(parts[16])
