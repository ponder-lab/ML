# The wala/ML#907 candidate: an elementwise add of two same-shaped numpy arrays, one of them scaled
# by a Python float. The result has the operands' shape, so a parameter fed this value should be as
# well typed as one fed the operands directly.
import numpy as np


def consume_operand(a):
    pass


def consume_scaled(a):
    pass


def consume_sum(a):
    pass


def consume_labels(a):
    pass


rng = np.random.RandomState(42)
logits = rng.uniform(size=(2, 20)).astype(np.float32)
labels = rng.permutation(np.eye(2, 20).T).T.astype(np.float32)

assert logits.shape == (2, 20)
consume_operand(logits)

assert labels.shape == (2, 20)
consume_labels(labels)

scaled = labels * 1000.0
assert scaled.shape == (2, 20)
consume_scaled(scaled)

combined = logits + labels * 1000.0
assert combined.shape == (2, 20)
consume_sum(combined)


def consume_array_sum(a):
    pass


def consume_int_scaled(a):
    pass


array_sum = logits + labels
assert array_sum.shape == (2, 20)
consume_array_sum(array_sum)

int_scaled = labels * 2
assert int_scaled.shape == (2, 20)
consume_int_scaled(int_scaled)


def consume_eye(a):
    pass


def consume_eye_t(a):
    pass


def consume_perm(a):
    pass


def consume_perm_t(a):
    pass


eye = np.eye(2, 20)
assert eye.shape == (2, 20)
consume_eye(eye)

eye_t = eye.T
assert eye_t.shape == (20, 2)
consume_eye_t(eye_t)

perm = rng.permutation(eye_t)
assert perm.shape == (20, 2)
consume_perm(perm)

perm_t = perm.T
assert perm_t.shape == (2, 20)
consume_perm_t(perm_t)
