# Witness for wala/ML#898, spelled as the crashing subject spells it.
#
# Three conditions must hold together for the type feed to be requested at all and to reach the
# receiver path: the input is taken from the receiver, so the position channel carries the receiver
# sentinel; the generator's seeded type leaves an axis unproven, since a fully-resolved seed
# returns before the feed is requested; and the frame has a caller invoke for the feed to walk.
#
# The second is what a hand-written `np.eye(2, 20).T` misses: it resolves, and returns early. The
# starred unpacking below leaves the extent unproven, which is what makes the receiver path
# reachable.
#
# Under all three the arity guard cannot screen the sentinel, since both sides of the comparison
# shift together, so the walk indexes the use list at -1 and throws instead of degrading.
import numpy as np


def consume(x):
    pass


logits_shape = (2, 20)
rng = np.random.default_rng(0)

labels = rng.permutation(np.eye(*logits_shape).T).T.astype(np.float32)

assert labels.shape == (2, 20)
assert labels.dtype == np.float32

consume(labels)
