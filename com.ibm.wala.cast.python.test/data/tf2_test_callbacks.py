import tensorflow as tf

def get_dataset(input_context):
  batch_size = input_context.get_per_replica_batch_size(2)
  return tf.data.Dataset.range(4).batch(batch_size)

global_batch_size = 2

strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])

input_context = tf.distribute.InputContext()
dist_dataset = strategy.distribute_datasets_from_function(get_dataset)
