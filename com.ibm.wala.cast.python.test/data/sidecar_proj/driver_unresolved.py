# The wala/ML#800 pair: an inferred Unresolved axis accepts a concrete annotated size (the
# annotation supplies exactly the fixed runtime integer Unresolved asserts the analysis could
# not compute), while a Dynamic axis keeps conflicting (it carries runtime-None evidence a
# concrete size would contradict).
import numpy as np
import tensorflow as tf


def consume(a):
    pass


def consume2(b):
    pass


labels = np.array([3, 1, 3, 2], dtype=int)
_, _, inv = np.unique(labels, return_index=True, return_inverse=True)
assert inv.dtype == np.int64
assert inv.shape == (4,)
consume(inv)

batch = tf.keras.Input(shape=(3,))
assert batch.shape.as_list() == [None, 3]
consume2(batch)
