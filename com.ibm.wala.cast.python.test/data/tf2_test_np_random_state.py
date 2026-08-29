# A seeded generator object reaches the same draws as the module-level surface
# (https://github.com/wala/ML/issues/827 modeled `np.random`; this is the `RandomState` form).
# The shape comes from `size=` exactly as it does for `np.random.uniform`.
import numpy as np
import tensorflow as tf


def consume_uniform(a):
    pass


def consume_normal(b):
    pass


def consume_cast(c):
    pass


rng = np.random.RandomState(42)

drawn = rng.uniform(size=(2, 20))
assert drawn.shape == (2, 20), drawn.shape
consume_uniform(tf.convert_to_tensor(drawn))

gaussian = rng.normal(size=(3, 4))
assert gaussian.shape == (3, 4), gaussian.shape
consume_normal(tf.convert_to_tensor(gaussian))

# The idiom the retrieval subjects use: draw, then narrow the dtype.
cast = rng.uniform(size=(2, 20)).astype(np.float32)
assert cast.shape == (2, 20), cast.shape
assert cast.dtype == np.float32, cast.dtype
consume_cast(tf.convert_to_tensor(cast))


def consume_module_cast(d):
    pass


# Control: the module-level surface with the same draw-then-narrow idiom. If this carries the same
# extra member, the extra member belongs to the narrowing rather than to the generator object.
module_cast = np.random.uniform(size=(2, 20)).astype(np.float32)
assert module_cast.shape == (2, 20), module_cast.shape
assert module_cast.dtype == np.float32, module_cast.dtype
consume_module_cast(tf.convert_to_tensor(module_cast))


def consume_raw_cast(e):
    pass


# Isolate the conversion: consume the narrowed array directly, without converting it.
raw_cast = np.random.uniform(size=(2, 20)).astype(np.float32)
assert raw_cast.shape == (2, 20), raw_cast.shape
consume_raw_cast(raw_cast)
