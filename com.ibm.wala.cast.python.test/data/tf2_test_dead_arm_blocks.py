# A call anywhere in an arm a constant guard never takes passes nothing to its callee: directly
# under the guard, after an `if` statement in the arm, in a loop in the arm, or after one.
import random

import tensorflow as tf


def under_guard(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def after_branch(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def after_one_armed_branch(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def in_for_loop(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def after_for_loop(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def in_while_loop(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def in_nested_loop(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def past_arm(x):
    assert x.shape == (4,) and x.dtype == tf.float32


class Model:
    def run(self, values, flag, pick):
        # The arm unpacks a third member, which the call that passes `False` does not hold.
        if flag:
            a, b, c = values
            under_guard(tf.abs(c))
            if pick:
                d = c
            else:
                d = c + 1.0
            after_branch(tf.abs(c))
            if pick:
                d = c * 2.0
            after_one_armed_branch(tf.abs(c))
            for i in range(2):
                in_for_loop(tf.abs(c))
            after_for_loop(tf.abs(c))
            n = 2
            while n > 0:
                in_while_loop(tf.abs(c))
                n -= 1
            for i in range(2):
                for j in range(2):
                    in_nested_loop(tf.abs(c))
        past_arm(tf.abs(values[0]))


x = tf.ones((4,))
y = tf.ones((5,))
z = tf.ones((2, 3))

model = Model()
model.run((x, y), False, len(x))
model.run((x, y, z), True, len(x))


def undecided_in_loop(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def undecided_after_branch(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


class Undecided:
    # The guard is no constant, so its arm may run.
    def run(self, values, flag, pick):
        if flag:
            a, b, c = values
            if pick:
                d = c
            else:
                d = c + 1.0
            undecided_after_branch(tf.abs(c))
            for i in range(2):
                undecided_in_loop(tf.abs(c))
        self.done = True


undecided = Undecided()
undecided.run((x, y, z), random.random() < 2, len(x))


def in_handler(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def after_handler(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


class Handled:
    # The handler is entered by the exception alone, and runs.
    def run(self, values, flag):
        a, b, c = values
        try:
            if flag:
                raise ValueError()
        except ValueError:
            in_handler(tf.abs(c))
        after_handler(tf.abs(c))


handled = Handled()
handled.run((x, y, z), True)


def in_else_arm(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def after_else_arm(x):
    assert x.shape == (4,) and x.dtype == tf.float32


class Other:
    # The guard is true at the one call, so the `else` arm never runs and what follows it does.
    def run(self, values, flag):
        if flag:
            a, b, c = values
        else:
            a, b = values
            c = tf.ones((7,))
            for i in range(2):
                in_else_arm(tf.abs(c))
        after_else_arm(tf.abs(values[0]))


other = Other()
other.run((x, y, z), True)
