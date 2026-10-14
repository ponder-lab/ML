# Test that a subclass of `tf.test.TestCase` has the summary shell of `TestCase` as its base, so
# the calls of its inherited methods have targets, while a subclass of `unittest.TestCase` and an
# override of an inherited method keep their own targets.
import unittest

import tensorflow as tf


class T(tf.test.TestCase):
    def evaluates(self, x):
        return self.evaluate(x)

    def asserts_close(self, x):
        self.assertAllClose(x, [1.0, 2.0])
        return tf.reduce_sum(x)

    def testRun(self):
        t = tf.constant([1.0, 2.0])
        self.evaluates(t)
        self.asserts_close(t)


class U(unittest.TestCase):
    def evaluate(self, x):
        return x

    def evaluates(self, x):
        return self.evaluate(x)

    def testRun(self):
        self.evaluates(tf.constant([1.0, 2.0]))


class V(tf.test.TestCase):
    def evaluate(self, x):
        return x

    def evaluates(self, x):
        return self.evaluate(x)

    def testRun(self):
        self.evaluates(tf.constant([1.0, 2.0]))


T().testRun()
U().testRun()
V().testRun()
