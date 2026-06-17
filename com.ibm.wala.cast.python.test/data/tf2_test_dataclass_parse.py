from dataclasses import dataclass

import tensorflow as tf


# Regression guard for wala/ML#205: a module containing a `@dataclass` definition
# must load and analyze without a front-end parse error. The dataclass is defined
# but not used in the dataflow; `f` receives a tensor directly, so its parameter
# types are recovered iff the module parsed. (Companion to `testModule68`/`69`,
# which guard the same for `NamedTuple`.)
@dataclass
class Holder:
    tensor: tf.Tensor


def f(x):
    pass


t = tf.ones([1, 2])
assert t.shape == (1, 2) and t.dtype == tf.float32
f(t)
