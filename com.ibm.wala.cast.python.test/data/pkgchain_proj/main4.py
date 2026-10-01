# Test https://github.com/wala/ML/issues/210.
import tensorflow as tf
import pkg as p
import pkg.sub as s

p.sub.f6(tf.ones((3, 14)))
s.f7(tf.ones((3, 15)))
