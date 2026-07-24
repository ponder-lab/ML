# Test the type-annotation sidecar (wala/ML#370): `np.load` is an opaque, content-dependent read
# the analysis cannot type; the sidecar supplies the type without touching this program.

import os

import numpy as np


def consume(x):
    pass


np.save("sidecar_tmp.npy", np.ones((4, 3), dtype=np.float32))
x = np.load("sidecar_tmp.npy")
assert x.shape == (4, 3)
assert x.dtype == np.float32
consume(x)
os.remove("sidecar_tmp.npy")
