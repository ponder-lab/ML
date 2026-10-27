# An allocator whose shape argument is an `int` of unknown value is given no rank (wala/ML#1159).
import pickle

import numpy as np


def consume_literal(m):
    pass


def consume_zeros(m):
    pass


def consume_length(m):
    pass


def consume_rows(m):
    pass


labels = np.vstack((pickle.loads(pickle.dumps(np.eye(4))), np.eye(4)))
literal = np.zeros(5)
assert literal.shape == (5,) and literal.dtype == np.float64
consume_literal(literal)
z = np.zeros(labels.shape[0])
assert z.shape == (8,) and z.dtype == np.float64
consume_zeros(z)
xs = [n for n in range(int("6"))]
by_length = np.zeros(len(xs))
assert by_length.shape == (6,) and by_length.dtype == np.float64
consume_length(by_length)
eye = np.eye(4)
rows = np.zeros(eye.shape[0])
assert rows.shape == (4,) and rows.dtype == np.float64
consume_rows(rows)
