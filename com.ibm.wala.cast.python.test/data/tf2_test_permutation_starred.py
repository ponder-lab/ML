# Reproduces the corpus construct for wala/ML#910 exactly: the eye dimensions arrive through a
# STARRED UNPACK of a tuple local (np.eye(*logits_shape)), not as literals. Isolates each step so the
# point where the shape is lost is unambiguous.
import numpy as np


def consume_eye_starred(a):
    pass


def consume_eye_starred_t(a):
    pass


def consume_perm_starred(a):
    pass


def consume_uniform_size(a):
    pass


rng = np.random.RandomState(42)
logits_shape = (2, 20)

# Positive control from the corpus: the tuple local read as size= resolves.
uniform_size = rng.uniform(size=logits_shape)
assert uniform_size.shape == (2, 20)
consume_uniform_size(uniform_size)

# Step 1: np.eye with a starred unpack of the tuple local.
eye_starred = np.eye(*logits_shape)
assert eye_starred.shape == (2, 20)
consume_eye_starred(eye_starred)

# Step 2: its transpose.
eye_starred_t = eye_starred.T
assert eye_starred_t.shape == (20, 2)
consume_eye_starred_t(eye_starred_t)

# Step 3: the permutation of that transpose (the exact corpus argument form).
perm_starred = rng.permutation(np.eye(*logits_shape).T)
assert perm_starred.shape == (20, 2)
consume_perm_starred(perm_starred)
