# Witness for wala/ML#810's safety residual, sidecar form. `t` is a top placeholder for inference
# (a `tf.constant` of an opaque `np.load`) that the sidecar annotates as `(4, 3)` float32; that type
# reaches `g`'s parameter only through the type analysis overlay. Inside the decorated body TensorFlow
# relaxes the static shape to the signature, `(None, 3)`, while the dtype stays float32, declared by the
# signature and by the annotation alike. Both are asserted inside the body.

import os

import numpy as np
import tensorflow as tf

sig = [tf.TensorSpec(shape=(None, 3), dtype=tf.float32)]


def consume(t):
    pass


def consume_pinned(a):
    pass


@tf.function(input_signature=sig)
def g(a):
    assert a.shape.as_list() == [None, 3]
    assert a.dtype == tf.float32
    consume_pinned(a)
    return a


np.save("sidecar_signature_pin_tmp.npy", np.ones((4, 3), dtype=np.float32))
raw = np.load("sidecar_signature_pin_tmp.npy")
t = tf.constant(raw)
assert t.shape == (4, 3)
assert t.dtype == tf.float32
consume(t)
out = g(t)
assert out.shape == (4, 3)
os.remove("sidecar_signature_pin_tmp.npy")
