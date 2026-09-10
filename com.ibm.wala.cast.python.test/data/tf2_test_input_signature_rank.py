# A `tf.function` with a declared `input_signature`. The declaration pins the parameter's RANK and
# dtype; the extents are `None`, so rank 2 is the recoverable fact and concrete extents are not.
import numpy as np
import tensorflow as tf

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Inputs"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Targets"),
]


def consume_declared(a):
    pass


def consume_helper(a):
    pass


def helper(seq):
    consume_helper(seq)
    return tf.cast(tf.math.equal(seq, 0), tf.float32)


@tf.function(input_signature=train_step_signature)
def train_step(inputs, targets):
    consume_declared(inputs)
    return helper(inputs)


out = train_step(tf.ones((4, 7), dtype=tf.int32), tf.ones((4, 7), dtype=tf.int32))
assert out.shape == (4, 7)


def consume_opaque_declared(a):
    pass


def consume_opaque_helper(a):
    pass


def opaque_helper(seq):
    consume_opaque_helper(seq)
    return tf.cast(tf.math.equal(seq, 0), tf.float32)


@tf.function(input_signature=train_step_signature)
def opaque_step(inputs, targets):
    consume_opaque_declared(inputs)
    return opaque_helper(inputs)


# The condition the subject is in: the ARGUMENT is unresolvable, so the declaration is the only
# source of a rank. A permutation of a transpose loses its rank for the analysis (wala/ML#910).
rng = np.random.RandomState(0)
opaque = tf.constant(rng.permutation(np.ones((7, 4)).T), dtype=tf.int32)
assert opaque.shape == (4, 7)

out2 = opaque_step(opaque, opaque)
assert out2.shape == (4, 7)


# A SECOND decorated function with a DIFFERENT declared rank, also called on an unresolvable
# argument. The two declarations differ (rank 2 vs rank 3), so a recognizer that keyed on anything
# shared — the signature list, or "some input_signature in the file" — would hand one function the
# other's rank. Under per-function keying each gets its own, which rank 2 vs rank 3 makes unmissable.
other_signature = [
    tf.TensorSpec(shape=(None, None, None), dtype=tf.int32, name="Volume"),
]


def consume_other(a):
    pass


@tf.function(input_signature=other_signature)
def other_step(x):
    consume_other(x)
    return x


opaque3 = tf.constant(rng.permutation(np.ones((4, 5, 6)).T), dtype=tf.int32)
assert opaque3.shape == (6, 5, 4)

out3 = other_step(opaque3)
assert out3.shape == (6, 5, 4)


# The CONTROL for the collision case below: a decorated METHOD with a single signature and no name
# clash. A method's decorator is not applied in IR, so this proves the recognizer fires on a method
# at all — before the collision test asks whether it fires AND resolves the right binding.
simple_method_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


def consume_simple_method(a):
    pass


class SimpleHolder:
    @tf.function(input_signature=simple_method_signature)
    def run(self, a):
        consume_simple_method(a)
        return a


out5 = SimpleHolder().run(opaque)
assert out5.shape == (4, 7)


# A signature name at MODULE scope (two specs, ranks 2 and 3) coexisting with a same-named INSTANCE
# ATTRIBUTE (one spec, rank 1), on a decorated METHOD. Python resolves the decorator's bare name to
# module scope, so the recognizer must read the two-spec module binding and index the second
# parameter into it — not the one-spec attribute, which would give parameter `b` nothing and `a` the
# wrong rank. Reproduces the gpt2_model.py collision, and exercises a decorated method (self offset).
collision_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
]


def consume_collide_a(a):
    pass


def consume_collide_b(b):
    pass


class Collider:
    def __init__(self):
        self.collision_signature = [tf.TensorSpec(shape=(None,), dtype=tf.int32)]

    @tf.function(input_signature=collision_signature)
    def step(self, a, b):
        consume_collide_a(a)
        consume_collide_b(b)
        return a


_collider = Collider()
out4 = _collider.step(opaque, opaque3)
assert out4.shape == (4, 7)


# A module name REBOUND before the decorated class — module-versus-module, the sibling of the
# collision above. Python takes whichever binding is live when the class body executes (the second,
# rank 3, here), but a name scan over the module body sees BOTH and cannot tell which dominates the
# decoration site. So the recognizer must DECLINE rather than guess, leaving the parameter rankless.
rebound_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int32)]
rebound_signature = [tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)]


def consume_rebound(a):
    pass


class ReboundHolder:
    @tf.function(input_signature=rebound_signature)
    def run(self, a):
        consume_rebound(a)
        return a


out6 = ReboundHolder().run(opaque3)
assert out6.shape == (6, 5, 4)
