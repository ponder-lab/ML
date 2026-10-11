# Test https://github.com/wala/ML/issues/1040: a parameter that `None` reaches, and that receives
# tensors in another context, has `None` among its types.
import tensorflow as tf


def optional(x, mask=None):
    if mask is None:
        return x
    return x * mask


def explicit(x):
    if x is None:
        return None
    return x + 1


def build_mask(inputs, sequence_length=None):
    if sequence_length is None:
        return tf.ones_like(inputs)
    return tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1])


def forward(inputs, sequence_length=None):
    return build_mask(inputs, sequence_length=sequence_length)


def never_tensor(flag=None):
    return flag


def always_tensor(x):
    return x * 2


a = optional(tf.ones([2]), tf.ones([2]))
assert a.shape == (2,)
b = optional(tf.ones([2]))
assert b.shape == (2,)

c = explicit(tf.ones([3]))
assert c.shape == (3,)
assert explicit(None) is None

d = forward(tf.ones([4, 5]), tf.constant([5, 3, 2, 1]))
assert d.shape == (4, 5)
e = forward(tf.ones([4, 5]))
assert e.shape == (4, 5)

assert never_tensor() is None
assert never_tensor(None) is None

f = always_tensor(tf.ones([2]))
assert f.shape == (2,)


def after_loop(x):
    return x * 2


for item in [tf.ones([2]), tf.ones([2])]:
    after_loop(item)
g = after_loop(item)
assert g.shape == (2,)


def maybe_bound(x):
    return x * 2


if tf.reduce_sum(tf.ones([2])) > 0:
    bound = tf.ones([2])
h = maybe_bound(bound)
assert h.shape == (2,)


def returned(x):
    if x is None:
        return None
    return x + 1


def maybe_none(flag):
    if flag:
        return None
    return tf.ones([2])


i = returned(tf.ones([2]))
assert i.shape == (2,)
assert returned(maybe_none(True)) is None


class Holder:
    def __init__(self, mask=None):
        self.mask = mask
        self.missing = None

    def apply(self, x):
        return from_field(x, self.missing)


def from_field(x, mask):
    if mask is None:
        return x
    return x * mask


j = from_field(tf.ones([2]), tf.ones([2]))
assert j.shape == (2,)
k = Holder().apply(tf.ones([2]))
assert k.shape == (2,)


def from_entry(x, mask):
    if mask is None:
        return x
    return x * mask


entries = {"mask": None}
m1 = from_entry(tf.ones([2]), tf.ones([2]))
assert m1.shape == (2,)
m2 = from_entry(tf.ones([2]), entries["mask"])
assert m2.shape == (2,)


def from_phi(x, mask):
    if mask is None:
        return x
    return x * mask


chosen = None if tf.reduce_sum(tf.ones([2])) > 5 else tf.ones([2])
n1 = from_phi(tf.ones([2]), tf.ones([2]))
assert n1.shape == (2,)
n2 = from_phi(tf.ones([2]), chosen)
assert n2.shape == (2,)


def from_star(x, mask):
    if mask is None:
        return x
    return x * mask


o1 = from_star(tf.ones([2]), tf.ones([2]))
assert o1.shape == (2,)
o2 = from_star(*[tf.ones([2]), None])
assert o2.shape == (2,)


def guarded_inner(mask):
    return mask * 2


def guarded_outer(x, mask=None):
    if mask is not None:
        x = x + guarded_inner(mask)
    return x


def else_inner(mask):
    return mask * 2


def else_outer(x, mask=None):
    if mask is None:
        return x
    else:
        return x + else_inner(mask)


def truth_inner(mask):
    return mask * 2


def truth_outer(x, mask=None):
    if mask:
        return x + truth_inner(mask)
    return x


p1 = guarded_outer(tf.ones([2]), tf.ones([2]))
assert p1.shape == (2,)
p2 = guarded_outer(tf.ones([2]))
assert p2.shape == (2,)
q1 = else_outer(tf.ones([2]), tf.ones([2]))
assert q1.shape == (2,)
q2 = else_outer(tf.ones([2]))
assert q2.shape == (2,)
r1 = truth_outer(tf.ones([2]), tf.constant(1.0))
assert r1.shape == (2,)
r2 = truth_outer(tf.ones([2]))
assert r2.shape == (2,)


def kernel_sink(kernel):
    return kernel * 2


class Built(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = tf.ones([2])

    def call(self, x):
        return x + kernel_sink(self.kernel)


s1 = Built()(tf.ones([2]))
assert s1.shape == (2,)


def scaled(x, scale=1.0):
    return x * scale


u1 = scaled(tf.ones([2]), tf.ones([2]))
assert u1.shape == (2,)
u2 = scaled(tf.ones([2]))
assert u2.shape == (2,)
