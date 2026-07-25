# Faithful loader-shape fixture (wala/ML#772): the annotated opaque read is conditionally
# reassigned through a slice inside its method, collected by a list comprehension in another
# method, and batched through `np.array`. Mirrors the in-vivo chain the plain append fixture
# (driver_feed.py) simplifies away.

import os
import random

import numpy as np


class Data:
    def __init__(self, files):
        self.files = files

    def _get_seq(self, fname, max_length=None):
        data = np.load(fname)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0, len(data) - max_length)
                data = data[start : start + max_length]
            else:
                data = np.append(data, 0)
        return data

    def batch(self, batch_size, length):
        batch_files = random.sample(self.files, k=batch_size)
        batch_data = [self._get_seq(file, length) for file in batch_files]
        return np.array(batch_data)


def consume(m):
    pass


np.save("sidecar_feed2_tmp.npy", np.arange(8, dtype=np.int64))
d = Data(["sidecar_feed2_tmp.npy", "sidecar_feed2_tmp.npy"])
m = d.batch(2, 4)
assert m.dtype == np.int64
consume(m)
os.remove("sidecar_feed2_tmp.npy")
