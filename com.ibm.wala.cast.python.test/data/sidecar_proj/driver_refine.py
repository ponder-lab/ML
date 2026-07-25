# Axis-level refinement fixture (wala/ML#771): values inference types only partially. The
# sidecar fills exactly the unknown axes: both axes of `t` (a top placeholder from `tf.constant`
# of an opaque read) and only the shape of `z` (whose float32 dtype inference already knows).

import os

import numpy as np
import tensorflow as tf


def consume(t):
    pass


def consume2(z):
    pass


np.save("sidecar_refine_tmp.npy", np.ones((4, 3), dtype=np.float32))
raw = np.load("sidecar_refine_tmp.npy")
t = tf.constant(raw)
assert t.shape == (4, 3)
assert t.dtype == tf.float32
consume(t)
dims = raw.shape
z = tf.zeros(dims)
assert z.shape == (4, 3)
assert z.dtype == tf.float32
consume2(z)
os.remove("sidecar_refine_tmp.npy")
