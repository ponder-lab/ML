# From https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function.

import tensorflow as tf

global_batch_size = 16
strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])


def dataset_fn(input_context):
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(64).batch(16)
  dataset = dataset.shard(
      input_context.num_input_pipelines, input_context.input_pipeline_id)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2)  # This prefetches 2 batches per device.
  return dataset


dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)
print(dist_dataset)
