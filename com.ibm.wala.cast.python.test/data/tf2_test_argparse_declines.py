import argparse
import random

import numpy as np

# The decline arms of the argparse-default chase (wala/ML#852) and the window fold's step guard
# (wala/ML#851), each observed through the comprehension-batcher carrier at the subject's own
# geometry: a module-level size read off the namespace, carried into a helper's `random.sample`
# lexically. The driver feeds explicit argv values so every arm RUNS with a real integer, while
# the statically visible default is the unrecoverable one; the runtime shapes below therefore
# deliberately differ from the analysis expectations, which pin the honest unknown.


def consume_control(a):
    pass


def consume_none_default(b):
    pass


def consume_no_default(c):
    pass


def consume_string_default(d):
    pass


def consume_twins(e):
    pass


def consume_retargeted(f):
    pass


def consume_dest_direct(g):
    pass


def consume_stepped(h):
    pass


def consume_short_option(i):
    pass


def consume_dynamic_option(j):
    pass


def consume_literal_bounds(m):
    pass


def consume_positional_k(n):
    pass


parser = argparse.ArgumentParser()

parser.add_argument("--good_size", default=2, type=int)
parser.add_argument("--none_size", default=None, type=int)
parser.add_argument("--bare_size", type=int)
parser.add_argument("--label_size", default="six", type=int)
parser.add_argument("--twin-size", default=3, type=int)
parser.add_argument("--twin_size", default=4, type=int)
parser.add_argument("--alpha", dest="beta_size", default=3, type=int)
parser.add_argument("--beta_size", default=5, type=int)
parser.add_argument("--routed", dest="routed_size", default=7, type=int)
parser.add_argument("-q", default=9, type=int)
dyn_option = "--dyn" + "_size"
parser.add_argument(dyn_option, default=11, type=int)

args = parser.parse_args(["--none_size", "2", "--bare_size", "2", "--label_size", "2"])

good_size = args.good_size
none_size = args.none_size
bare_size = args.bare_size
label_size = args.label_size
twin_size = args.twin_size
beta_size = args.beta_size
routed_size = args.routed_size
q = args.q
dyn_size = args.dyn_size

pool = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]


def window(f):
    return [0, 0, 0]


def batch_control():
    return np.array([window(f) for f in random.sample(pool, k=good_size)])


def batch_none():
    return np.array([window(f) for f in random.sample(pool, k=none_size)])


def batch_bare():
    return np.array([window(f) for f in random.sample(pool, k=bare_size)])


def batch_label():
    return np.array([window(f) for f in random.sample(pool, k=label_size)])


def batch_twins():
    return np.array([window(f) for f in random.sample(pool, k=twin_size)])


def batch_retargeted():
    return np.array([window(f) for f in random.sample(pool, k=beta_size)])


def batch_dest():
    return np.array([window(f) for f in random.sample(pool, k=routed_size)])


def stepped(f):
    data = list(range(50))
    start = 1
    return data[start : start + good_size : 2]


def batch_stepped():
    return np.array([stepped(f) for f in random.sample(pool, k=2)])


def batch_short():
    return np.array([window(f) for f in random.sample(pool, k=q)])


def batch_dynamic():
    return np.array([window(f) for f in random.sample(pool, k=dyn_size)])


def clipped(f):
    data = list(range(50))
    return data[2:6]


def batch_clipped():
    return np.array([clipped(f) for f in random.sample(pool, k=2)])


def batch_positional():
    return np.array([window(f) for f in random.sample(pool, 2)])


# The positive control: a plain integer literal default resolves, which is what makes the
# unresolved pins below measurements rather than mirrors of the no-inference floor.
control = batch_control()
assert control.shape == (2, 3), control.shape
consume_control(control)

# default=None: no literal to recover.
none_batch = batch_none()
assert none_batch.shape == (2, 3), none_batch.shape
consume_none_default(none_batch)

# No default at all.
bare_batch = batch_bare()
assert bare_batch.shape == (2, 3), bare_batch.shape
consume_no_default(bare_batch)

# A string default: argparse coerces only command-line values, so the default is not the
# integer the type annotation suggests.
label_batch = batch_label()
assert label_batch.shape == (2, 3), label_batch.shape
consume_string_default(label_batch)

# Two spellings deriving one destination with different defaults: registration order decides
# the runtime value (the FIRST registration's default sticks), so the static answer declines
# rather than picks.
twin_batch = batch_twins()
assert twin_batch.shape == (3, 3), twin_batch.shape
consume_twins(twin_batch)

# A `dest=` retarget beside a derived twin: two arguments write `beta_size` through different
# routes and disagree, so the static answer declines; the first registration gives 3 at runtime.
retargeted_batch = batch_retargeted()
assert retargeted_batch.shape == (3, 3), retargeted_batch.shape
consume_retargeted(retargeted_batch)

# A `dest=` with no competing writer is argparse's authoritative destination and resolves.
dest_batch = batch_dest()
assert dest_batch.shape == (7, 3), dest_batch.shape
consume_dest_direct(dest_batch)

# The window fold's step guard: a non-unit step is not the contract `stop - start` computes, so
# the window degrades while the arity still resolves.
stepped_batch = batch_stepped()
assert stepped_batch.shape == (2, 1), stepped_batch.shape
consume_stepped(stepped_batch)

# A short option alone: its destination is the option name with the dash stripped, and the
# literal default resolves through it.
short_batch = batch_short()
assert short_batch.shape == (9, 3), short_batch.shape
consume_short_option(short_batch)

# A non-literal option string: the call cannot match any read by name, so the read of the
# attribute it creates at runtime has no candidate and stays unresolved.
dynamic_batch = batch_dynamic()
assert dynamic_batch.shape == (11, 3), dynamic_batch.shape
consume_dynamic_option(dynamic_batch)

# Literal slice bounds subtract: the window is 6 - 2 whatever the sliced data is.
clipped_batch = batch_clipped()
assert clipped_batch.shape == (2, 4), clipped_batch.shape
consume_literal_bounds(clipped_batch)

# `random.sample`'s count passed positionally rather than as `k=`.
positional_batch = batch_positional()
assert positional_batch.shape == (2, 3), positional_batch.shape
consume_positional_k(positional_batch)
