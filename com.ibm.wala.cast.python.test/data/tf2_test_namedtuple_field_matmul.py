"""Exercises dtype recovery for ``tf.linalg.matmul`` on a ``NamedTuple`` field threaded in as a parameter (https://github.com/wala/ML/issues/570).

``Inp`` is constructed in the caller and passed into ``layer``, which reads ``inp.x`` (a ``NamedTuple`` field) and feeds it to ``tf.linalg.matmul``. The matmul input's dtype is recovered (``float32``) by reading the field off the instance in the heap, even though the field read has no points-to set at the read site. This mirrors the minimal form of the ``gcn_proj`` ``GraphConvolution.call`` inner chain. Shape stays ⊤ (dtype is the load-bearing axis here; shape recovery is follow-up).
"""

from typing import NamedTuple, List

import numpy as np
import tensorflow as tf


class Inp(NamedTuple):
    x: tf.Tensor
    rest: List


def consume(t):
    pass


def layer(inp):
    h = tf.linalg.matmul(inp.x, inp.x)
    consume(h)
    return h


a = tf.constant(np.ones((4, 4), dtype=np.float32))
assert a.shape == (4, 4) and a.dtype == tf.float32
layer(Inp(a, []))
