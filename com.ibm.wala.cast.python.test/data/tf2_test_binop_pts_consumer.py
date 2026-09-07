# Witness for wala/ML#805: a binary operator's result has no allocation site, so its points-to set
# is empty and every consumer that does a heap read off it sees nothing. Here `shifted` is the
# result of an operator over a real `np.array` allocation, and the consumers below reach the value
# through the heap rather than by walking the SSA chain.
import numpy as np


def consume(x):
    pass


def consume_row(x):
    pass


rows = np.array([[1, 2, 3], [4, 5, 6]])
shifted = rows - 1

# A subscript receiver: resolving this needs `shifted` to have a points-to set.
row = shifted[0]
consume_row(row)
assert row.shape == (3,)

consume(shifted)
assert shifted.shape == (2, 3)


def consume_control(x):
    pass


# Control: the same subscript with NO operator between the allocation and the read. If this also
# fails to type, the subscript path is the gap rather than wala/ML#805's missing allocation.
control_row = rows[1]
consume_control(control_row)
assert control_row.shape == (3,)
