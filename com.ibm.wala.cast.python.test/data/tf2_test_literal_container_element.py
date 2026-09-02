# Third witness for wala/ML#867's predicate: the same dual-channel geometry as
# tf2_test_none_container_element.py, but the leading element's non-tensor member
# is an ORDINARY literal rather than None. The element field key carries tensor
# may-state beside a non-null constant, and the None-possibility predicate must
# stay false: it flags None specifically, not any constant.
import numpy as np


def read():
    a = 1.5
    b = np.array([[5.0], [6.0]])
    names = ["node_attributes"]
    if "node_attributes" in names:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
    return [a, b]


def cat(seq):
    return np.concatenate(seq, axis=-1)


x = read()
r = cat(x)
assert r.shape == (2, 3)
