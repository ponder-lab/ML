# `np.vstack` joins arrays along their first axis, raising each array of rank below 2 to a row
# first (wala/ML#1159).
import numpy as np


def consume_matrices(t):
    pass


def consume_rows(t):
    pass


matrices = np.vstack((np.zeros((2, 3)), np.ones((4, 3))))
assert matrices.shape == (6, 3) and matrices.dtype == np.float64
consume_matrices(matrices)

rows = np.vstack((np.zeros(3), np.ones(3)))
assert rows.shape == (2, 3) and rows.dtype == np.float64
consume_rows(rows)
