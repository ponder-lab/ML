# Test https://github.com/wala/ML/issues/210.
import tensorflow as tf
import qkg

qkg.sub.k(tf.ones((3, 12)))
qkg.sub.Scaler()(tf.ones((3, 13)))
