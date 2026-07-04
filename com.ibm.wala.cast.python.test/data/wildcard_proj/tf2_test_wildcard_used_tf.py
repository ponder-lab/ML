# Probe for wala/ML#684: `tf` reached through `from helpers_used import *`, where the exporting
# module also reads `tf` inside one of its own functions. The subject's `custom/layers.py` has this
# shape (its layer classes read `tf`), while the `helpers.py` probes leave the binding untouched;
# the exposure is the fixture-scale divergence between the two.
from helpers_used import *


def consume(t):
    pass


x = tf.ones((4, 4))
consume(x)
assert x.shape == (4, 4)
