import numpy as np


def consume_array_arm(a):
    pass


def consume_int_arm(b):
    pass


def consume_module_arm(c):
    pass


# `permutation` shuffles along the FIRST axis only, so an array argument's shape survives it
# unchanged. The two arms disagree, which is why they are pinned separately: an integer argument
# yields a shuffled `arange`, rank one of that length, where passing the integer's own scalar shape
# through would report a scalar the runtime never produces.
rng = np.random.RandomState(42)

permuted = rng.permutation(np.eye(2, 20))
assert permuted.shape == (2, 20), permuted.shape
consume_array_arm(permuted)

indices = rng.permutation(10)
assert indices.shape == (10,), indices.shape
assert indices.dtype == np.int64, indices.dtype
consume_int_arm(indices)

# The module-level surface carries the same operation and must agree with the generator object's.
module_permuted = np.random.permutation(np.eye(3, 7))
assert module_permuted.shape == (3, 7), module_permuted.shape
consume_module_arm(module_permuted)
