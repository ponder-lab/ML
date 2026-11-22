# A function that returns True on one path and falls off its end on the others, so a guard on its
# result is taken only sometimes, and the code after the guard's `break` runs.
import tensorflow as tf

calls = {"after": 0, "loop": 0}


def consume_after_guard(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32
    calls["after"] += 1


def consume_in_loop_after_guard(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32
    calls["loop"] += 1


def consume_before_guard(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def should_stop(value):
    if value > 3:
        return True


for step in range(5):
    consume_before_guard(tf.ones([2, 3]))
    if should_stop(step):
        break
    consume_after_guard(tf.ones([2, 3]))
    for item in [1]:
        consume_in_loop_after_guard(tf.ones([2, 3]))

assert calls == {"after": 4, "loop": 4}
