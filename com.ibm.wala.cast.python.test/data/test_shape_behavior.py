import tensorflow as tf

v1 = tf.Variable([[1, 2], [3, 4]], shape=[None, 2])
print(v1.shape)
print(v1.shape.as_list())
