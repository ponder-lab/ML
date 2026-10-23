# Test `ContainerElements` on lists a library summary or operator allocates, and on lists the
# program mutates in ways the analysis does not record among their positioned elements. Each sink
# receives one container; a reported length must be the asserted run-time one.
import bisect
import heapq
import operator
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_unbound_append(c):
    assert len(c) == 2


def sink_getattr_append(c):
    assert len(c) == 2


def sink_operator_iadd(c):
    assert len(c) == 2


def sink_heappush(c):
    assert len(c) == 2


def sink_insort(c):
    assert len(c) == 2


def sink_empty_heappush(c):
    assert len(c) == 1


def sink_sorted_key_lambda(c):
    assert len(c) == 3


def sink_map_lambda(c):
    assert len(c) == 3


def sink_tuple_of_appended(c):
    assert len(c) == 2


def sink_list_of_appended(c):
    assert len(c) == 2


def sink_copy_of_appended(c):
    assert len(c) == 2


def sink_imul(c):
    assert len(c) == 4


def sink_slice_insert(c):
    assert len(c) == 3


def sink_attr_append(c):
    assert len(c) == 2


def sink_bound_method(c):
    assert len(c) == 2


def sink_comp_mutation(c):
    assert len(c) == 3


def sink_unbound_extend_callee(c):
    assert len(c) == 3


def sink_split(c):
    assert len(c) == 3


def sink_list_range(c):
    assert len(c) == 3


def sink_sorted_set(c):
    assert len(c) == 3


def sink_unstack(c):
    assert len(c) == 2


def sink_zip_tuple(c):
    assert len(c) == 2


def sink_range_obj(c):
    assert len(c) == 3


def sink_split_tf(c):
    assert len(c) == 4


def sink_items(c):
    assert len(c) == 3


def sink_enumerate_list(c):
    assert len(c) == 3


def sink_crop_size(c):
    assert len(c) == 3


def extend_through_unbound(l):
    list.extend(l, [y, y])


unbound = [x]
list.append(unbound, y)
sink_unbound_append(unbound)

ga = [x]
getattr(ga, "append")(y)
sink_getattr_append(ga)

oi = [x]
operator.iadd(oi, [y])
sink_operator_iadd(oi)

hp = [1]
heapq.heappush(hp, 2)
sink_heappush(hp)

ins = [1]
bisect.insort(ins, 2)
sink_insort(ins)

eh = []
heapq.heappush(eh, 1)
sink_empty_heappush(eh)

sk = [x]
sorted([1, 2], key=lambda v: sk.append(y) or v)
sink_sorted_key_lambda(sk)

ml = [x]
list(map(lambda v: ml.append(v), [y, y]))
sink_map_lambda(ml)

ta = [x]
ta.append(y)
sink_tuple_of_appended(tuple(ta))

la = [x]
la.append(y)
sink_list_of_appended(list(la))

ca = [x]
ca.append(y)
sink_copy_of_appended(ca.copy())

im = [x, y]
im *= 2
sink_imul(im)

si = [x, y]
si[1:1] = [x]
sink_slice_insert(si)


class Holder:
    def __init__(self):
        self.items = [x]


h = Holder()
h.items.append(y)
sink_attr_append(h.items)

bm = [x]
add = bm.append
add(y)
sink_bound_method(bm)

cm = [x]
[cm.append(v) for v in [y, y]]
sink_comp_mutation(cm)

ue = [x]
extend_through_unbound(ue)
sink_unbound_extend_callee(ue)

sink_split("a,b,c".split(","))
sink_list_range(list(range(3)))
sink_sorted_set(sorted({3, 1, 2}))
sink_unstack(tf.unstack(tf.ones([2, 3])))

for zt in zip([x, x], [y, y]):
    sink_zip_tuple(zt)
sink_range_obj(range(3))
sink_split_tf(tf.split(tf.ones([8, 3]), 4))

dct = {"a": x, "b": y, "c": x}
sink_items(dct.items())
sink_enumerate_list(list(enumerate([x, y, x])))

# A list a TensorFlow function reads, which keeps its length.
size = [1, 2, 3]
tf.image.random_crop(tf.ones([4, 4, 3]), size)
sink_crop_size(size)
