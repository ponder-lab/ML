# Fixture for wala/ML#1043: an annotation supplying a concrete extent for the axis a `-1` reshape
# leaves as `?`.
#
# `read` reshapes a flat array of values the analysis cannot count into rows of 7. The `-1` axis is
# the row count, which inference holds as `?` because the input's length is not computed. A user
# who knows the file their program reads can record that count, and nothing in `?` disagrees with
# it: `?` says only that the extent was not computed. `read_named` is the same read with an entry
# that names the row count instead, which keeps the `?`.
import numpy as np


def consume(x):
    pass


def consume_named(x):
    pass


def read(values):
    rows = np.array(values, dtype=np.float32)
    x = np.reshape(rows, [-1, 7])
    consume(x)
    return x


def read_named(values):
    rows = np.array(values, dtype=np.float32)
    x = np.reshape(rows, [-1, 7])
    consume_named(x)
    return x


out = read([0.0] * 21)
assert out.shape == (3, 7)
assert out.dtype == np.float32

named = read_named([0.0] * 14)
assert named.shape == (2, 7)
