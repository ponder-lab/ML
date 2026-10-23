# Test `ContainerElements` on sets grown by `add`, `|=` and `update`. Each sink receives one set.
import tensorflow as tf

# A tensor is hashable, and so a set's element, only with tensor equality off.
tf.compat.v1.disable_tensor_equality()

x = tf.ones([2, 3])
z = tf.zeros([4], tf.int32)


def sink_empty_added(c):
    assert len(c) == 1 and x in c


def sink_added(c):
    assert len(c) == 2 and x in c


def sink_ior(c):
    assert len(c) == 2 and x in c


def sink_ior_alias(c):
    assert len(c) == 2 and x in c


def sink_updated(c):
    assert len(c) == 2 and x in c


empty_added = set()
empty_added.add(x)
sink_empty_added(empty_added)

added = {z}
added.add(x)
sink_added(added)

ior = {z}
ior |= {x}
sink_ior(ior)

ior_source = {z}
ior_alias = ior_source
ior_source |= {x}
sink_ior_alias(ior_alias)

updated = {z}
updated.update([x])
sink_updated(updated)
