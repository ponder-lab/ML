# Witness for wala/ML#867: a container whose leading element is None on every
# reachable runtime path, while a statically feasible branch assigns an ndarray
# behind a data-dependent guard. The container's leading element field therefore
# carries tensor MAY-state and a None constant on the same pointer key.
import numpy as np


def read():
    a = None
    b = None
    names = ["node_labels"]
    if "node_attributes" in names:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
    if "node_labels" in names:
        b = np.array([[5.0], [6.0]])
    return [a, b]


def cat(seq):
    seq = [x for x in seq if x is not None]
    return np.concatenate(seq, axis=-1)


x = read()
r = cat(x)
assert r.shape == (2, 1)
