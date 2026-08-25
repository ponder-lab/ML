# wala/ML#835: `np.transpose` and `ndarray.transpose` were unmodeled, so a transposed array was
# untyped and everything downstream degraded.
import sys

import numpy as np


def consume_permuted(x):
    assert x.shape == (4, 2, 3)
    assert x.dtype == np.float32


def consume_reversed(x):
    assert x.shape == (4, 3, 2)
    assert x.dtype == np.float32


def consume_method(x):
    assert x.shape == (3, 2)
    assert x.dtype == np.float32


def consume_method_axes(x):
    # The method form with a positional `axes`: the argument must bind to `axes`, not to the
    # input slot; the receiver is the input.
    assert x.shape == (3, 2, 4)
    assert x.dtype == np.float32


def consume_negative(x):
    assert x.shape == (4, 2, 3)
    assert x.dtype == np.float32


def consume_none_axes(x):
    assert x.shape == (4, 3, 2)
    assert x.dtype == np.float32


def consume_unresolved(x):
    # The permutation depends on a runtime branch the analysis cannot fold, so it reports an
    # unresolved size per axis while preserving the rank; the runtime truth for this
    # configuration is the else arm's permutation.
    assert x.shape == (3, 4, 2)
    assert x.dtype == np.float32


def consume_mixed(x):
    # `axes` is `None` on one branch and a constant permutation on the other, so the analysis
    # reports both the reversal and the permutation; the runtime truth for this configuration
    # is the else arm's permutation.
    assert x.shape == (4, 2, 3)
    assert x.dtype == np.float32


def consume_chained(x):
    # The transpose result consumed by an elementwise op: the op's operand resolution reaches
    # the summary's allocation through producer delegation, the node-anchored dispatch route.
    assert x.shape == (4, 2, 3)
    assert x.dtype == np.float32


def consume_method_chained(x):
    assert x.shape == (3, 2)
    assert x.dtype == np.float32


a = np.zeros((2, 3, 4), dtype=np.float32)
consume_permuted(np.transpose(a, (2, 0, 1)))
consume_reversed(np.transpose(a))
consume_negative(np.transpose(a, (-1, 0, 1)))
consume_none_axes(np.transpose(a, None))

axes = (2, 0, 1) if len(sys.argv) > 99 else (1, 2, 0)
consume_unresolved(np.transpose(a, axes))

mixed_axes = None if len(sys.argv) > 99 else (2, 0, 1)
consume_mixed(np.transpose(a, mixed_axes))

b = np.ones((2, 3), dtype=np.float32)
consume_method(b.transpose())

c = np.zeros((2, 3, 4), dtype=np.float32)
consume_method_axes(c.transpose((1, 0, 2)))

consume_chained(np.transpose(a, (2, 0, 1)) + 1.0)
consume_method_chained(b.transpose() * 2.0)
