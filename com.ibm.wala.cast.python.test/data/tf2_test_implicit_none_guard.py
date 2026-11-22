# A function that returns True on one path and falls off its end on the others, so a guard on its
# result is taken only sometimes, and the code after the guard's `break` runs.
import tensorflow as tf

calls = {
    "after": 0,
    "loop": 0,
    "recursive": 0,
    "logged": 0,
    "log": 0,
    "bare": 0,
    "mode": 0,
}


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


def consume_never(t):
    raise AssertionError("never called")


def always(value):
    return True


def consume_never_looped(t):
    raise AssertionError("never called")


def always_looped(value):
    while True:
        return True


for step in range(2):
    if always_looped(step):
        continue
    consume_never_looped(tf.ones([2, 3]))

for step in range(2):
    if always(step):
        continue
    consume_never(tf.ones([2, 3]))


def consume_after_recursive_guard(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32
    calls["recursive"] += 1


def done(n):
    if n == 0:
        return True
    if done(n - 1):
        return True


def ping(n):
    if n <= 0:
        return True
    if pong(n - 1):
        return True


def pong(n):
    if ping(n):
        return True


for step in range(3):
    if done(step) and ping(step):
        continue
    consume_after_recursive_guard(tf.ones([2, 3]))

for step in range(5):
    consume_before_guard(tf.ones([2, 3]))
    if should_stop(step):
        break
    consume_after_guard(tf.ones([2, 3]))
    for item in [1]:
        consume_in_loop_after_guard(tf.ones([2, 3]))

assert (calls["after"], calls["loop"], calls["recursive"]) == (4, 4, 0)


def consume_never_in_mode(t):
    raise AssertionError("never called")


def always_in_mode(mode):
    if mode == "train":
        return True


for step in range(2):
    if always_in_mode("train"):
        continue
    consume_never_in_mode(tf.ones([2, 3]))


def consume_after_logged_guard(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32
    calls["logged"] += 1


def logged_stop(value):
    if value > 3:
        return True
    calls["log"] += 1


for step in range(5):
    if logged_stop(step):
        break
    consume_after_logged_guard(tf.ones([2, 3]))


def consume_after_bare_guard(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32
    calls["bare"] += 1


def bare_stop(value):
    if value > 3:
        return True
    return


for step in range(5):
    if bare_stop(step):
        break
    consume_after_bare_guard(tf.ones([2, 3]))


def consume_never_in_logged_mode(t):
    raise AssertionError("never called")


def always_in_logged_mode(mode):
    if mode == "train":
        return True
    calls["mode"] += 1


for step in range(2):
    if always_in_logged_mode("train"):
        continue
    consume_never_in_logged_mode(tf.ones([2, 3]))

assert calls["logged"] == 4 and calls["bare"] == 4 and calls["mode"] == 0
