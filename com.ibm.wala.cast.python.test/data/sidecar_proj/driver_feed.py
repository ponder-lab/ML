# Feed-through fixture (wala/ML#772): a sidecar annotation on an opaque read reaches an
# `np.array` result through the dtype feed. Mirrors the loader shape that motivated the issue:
# an opaque per-item read returned from a helper, accumulated in a list, batched through
# `np.array`. The list hop matters: the argument's own variable carries no dataflow state, so
# the feed must read the container's element keys.

import os

import numpy as np


def get_seq():
    data = np.load("sidecar_feed_tmp.npy")
    return data


def consume(m):
    pass


def consume2(b):
    pass


np.save("sidecar_feed_tmp.npy", np.ones((4, 3), dtype=np.int64))
batch_data = []
for _ in range(2):
    batch_data.append(get_seq())
m = np.array(batch_data)
assert m.dtype == np.int64
consume(m)
# An explicit dtype argument overrides the source's dtype at runtime, so when it is statically
# unresolvable the feed must decline rather than borrow the operand's dtype (wala/ML#774).
dt = np.load("sidecar_feed_tmp.npy").dtype
b = np.array(batch_data, dtype=dt)
assert b.dtype == np.int64
consume2(b)
os.remove("sidecar_feed_tmp.npy")
