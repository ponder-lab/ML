# Sibling witness to tf2_test_none_container_element.py (wala/ML#867): the same
# list-into-callee geometry with every element unconditionally an ndarray, so no
# None constant reaches the container's element field.
import numpy as np


def read():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0], [6.0]])
    return [a, b]


def cat(seq):
    return np.concatenate(seq, axis=-1)


x = read()
r = cat(x)
assert r.shape == (2, 3)
