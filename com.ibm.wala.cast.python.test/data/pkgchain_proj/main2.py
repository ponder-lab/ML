# Test https://github.com/wala/ML/issues/210.
import tensorflow as tf
from pkg import sub
import pkg.sub
import pkg.sub.mod as m

sub.f2(tf.ones((3, 8)))
pkg.sub.f3(tf.ones((3, 9)))
m.f4(tf.ones((3, 10)))
s = pkg.sub
s.f5(tf.ones((3, 11)))
