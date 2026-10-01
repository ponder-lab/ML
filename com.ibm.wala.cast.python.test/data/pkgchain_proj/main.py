# Test https://github.com/wala/ML/issues/210.
import tensorflow as tf
import pkg
from pkg.sub import g

pkg.sub.f(tf.ones((3, 4)))
g(tf.ones((3, 5)))
pkg.sub.mod.h(tf.ones((3, 6)))
pkg.sub.Scale()(tf.ones((3, 7)))
