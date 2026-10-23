# Test `ContainerElements` where the program deletes an element with a subscript `del`, whose target
# the front end drops: no list reports a length. Each sink receives one container; a reported length
# must be the asserted run-time one.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_callee_del(c):
    assert len(c) == 1


def sink_global_del(c):
    assert len(c) == 1


def sink_deep_del(c):
    assert len(c) == 1


def sink_dict_deep_del(c):
    assert len(c) == 1


def sink_closure_del(c):
    assert len(c) == 1


def sink_untouched(c):
    assert len(c) == 2


def sink_tuple(c):
    assert len(c) == 2


def delete_first(l):
    del l[0]


cd = [x, y]
delete_first(cd)
sink_callee_del(cd)

G = [x, y]


def del_global():
    del G[0]


del_global()
sink_global_del(G)


class Inner:
    def __init__(self):
        self.items = [x, y]


class Outer:
    def __init__(self):
        self.inner = Inner()


def del_deep(o):
    del o.inner.items[0]


outer = Outer()
del_deep(outer)
sink_deep_del(outer.inner.items)


def del_dict_deep(d):
    del d["a"]["b"][0]


dd = {"a": {"b": [x, y]}}
del_dict_deep(dd)
sink_dict_deep_del(dd["a"]["b"])


def make_closure_del():
    cl = [x, y]

    def inner():
        del cl[0]

    inner()
    return cl


sink_closure_del(make_closure_del())

# A list no `del` touches still reports none, since the deleted one is unknown; a tuple is immutable.
sink_untouched([x, y])
sink_tuple((x, y))
