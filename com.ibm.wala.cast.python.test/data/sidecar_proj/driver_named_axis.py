# Fixture for an annotation that names an axis whose extent varies with the input.
#
# `make` builds one row per label, so its leading axis is a different size for each call: the
# analysis records it as `Unresolved`, and an author can only name it. The second axis is a count
# computed from the loaded labels, which the author can state as a number. Naming the first axis
# must not stop the number from filling the second.
import numpy as np


def consume(x):
    pass


data = [np.array([0, 3, 1]), np.array([2, 0])]
depth = np.max([np.max(labels) for labels in data]) + 1


def make(labels):
    num = len(labels)
    x = np.zeros([num, depth], dtype=np.float32)
    x[list(range(num)), labels] = 1.0
    consume(x)
    return x


for labels in data:
    out = make(labels)
    assert out.shape == (len(labels), 4)
    assert out.dtype == np.float32
