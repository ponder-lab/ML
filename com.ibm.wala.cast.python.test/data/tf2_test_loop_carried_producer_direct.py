# Loop-carried producers read DIRECTLY through a generator, with no analysis run and so no worklist
# resolver installed (wala/ML#928). Each loop makes a variable depend on its own producer's result,
# so a generator reading the operand meets its own allocation: the producer self-recursion guards.
# The loops MUST be loops; written-out repetitions form a chain and never re-enter.
import tensorflow as tf


def consume_concat(x):
    pass


def consume_add(x):
    pass


def consume_tile(x):
    pass


# The tiling doubles the leading extent each pass: three iterations take (2, 3) to (16, 3), so the
# reachable set of `z` is (2, 3), (4, 3), (8, 3), (16, 3) and the final tiling is (32, 3). A guard
# that drops the self-referential member leaves a set of concrete members that excludes that value.
z = tf.ones((2, 3))
for _ in range(3):
    z = tf.tile(z, [2, 1])
assert z.shape == (16, 3)
final_tile = tf.tile(z, [2, 1])
assert final_tile.shape == (32, 3)
consume_tile(final_tile)

# The concatenation doubles the leading extent the same way, but its operand is a list, and the
# concatenation declines to an unknown shape when any operand member is missing, so it reads the
# same either way: a control, not a witness.
x = tf.ones((2, 3))
for _ in range(3):
    x = tf.concat([x, x], 0)
assert x.shape == (16, 3)
final = tf.concat([x, x], 0)
assert final.shape == (32, 3)
consume_concat(final)

# The addition keeps the shape, (2, 3) on every iteration, so the program's value is in the set
# either way and only the unknown mark distinguishes the two guard values here.
y = tf.ones((2, 3))
for _ in range(3):
    y = tf.add(y, 1.0)
assert y.shape == (2, 3)
consume_add(tf.add(y, 1.0))
