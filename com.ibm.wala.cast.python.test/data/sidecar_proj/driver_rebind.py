# Fixtures for wala/ML#890: a sidecar entry names a VARIABLE, and a variable can be bound more
# than once, so an entry describing one value is also judged against values it was never about.
#
# `reassigned` is the straight-line shape (instance one): the string binding is OVERWRITTEN on the
# only path, so it is not a terminal value of `v` at all, and an entry about the tensor should not
# be judged against it.
#
# `branched` is the exclusive-arms shape (instance two): BOTH bindings are terminal, each on its
# own path, so an entry agreeing with one genuinely disagrees with the other. That conflict is
# TRUE and must keep being reported; suppressing it would hide a correct statement about a
# reachable path.
import tensorflow as tf


def consume_reassigned(x):
    pass


def consume_branched(x):
    pass


def reassigned():
    v = tf.constant("abc")
    v = tf.zeros((4, 3))
    consume_reassigned(v)


def branched(flag):
    if flag:
        w = tf.zeros((4, 3))
    else:
        w = tf.zeros((4, 3), dtype=tf.int32)
    consume_branched(w)


reassigned()
branched(True)
