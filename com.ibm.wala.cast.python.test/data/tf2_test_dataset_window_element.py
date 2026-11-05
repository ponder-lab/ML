# Test https://github.com/wala/ML/issues/1201: `window` yields datasets whose own iteration yields
# the source elements, and for a tuple element a tuple of datasets, one per component.
import os
import tempfile

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
