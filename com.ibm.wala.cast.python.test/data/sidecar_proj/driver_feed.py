# Feed-through fixture (wala/ML#772): a sidecar annotation on an opaque read reaches an
# `np.array` result through the dtype feed. Mirrors the loader shape that motivated the issue:
# an opaque per-item read returned from a helper, batched through `np.array`.

import os

import numpy as np


def get_seq():
    data = np.load("sidecar_feed_tmp.npy")
    return data


def consume(m):
    pass


np.save("sidecar_feed_tmp.npy", np.ones((4, 3), dtype=np.int64))
batch_data = get_seq()
m = np.array(batch_data)
assert m.dtype == np.int64
consume(m)
os.remove("sidecar_feed_tmp.npy")
