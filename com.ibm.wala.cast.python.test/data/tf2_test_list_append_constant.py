# Reproduces wala/ML#668: appending a constant (an invariant-contents value) must not crash
# call-graph construction. The invoke's own argument processing records the constant's pointer
# key as implicitly represented; the append model must use the invariant-contents path instead
# of a raw constraint on that key. The tensor appended alongside still types through iteration.
import tensorflow as tf


def consume(t):
    pass


xs = []
xs.append(1)
xs.append(tf.ones((4, 4)))

for x in xs:
    consume(x)
