package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Base for shape-first random samplers &mdash; {@code tf.random.uniform}, {@code tf.random.normal},
 * {@code tf.random.truncated_normal} &mdash; which share the {@code (shape, ..., dtype, seed,
 * name)} signature: the explicit shape at position 0 (inherited from {@link TensorTypeAllocator})
 * and the dtype at position 3. Each concrete sampler is type-inferred identically (shape from the
 * shape argument, default {@code float32} dtype), so they extend this base rather than one another,
 * replacing the {@code TruncatedNormal extends Normal extends Uniform} chain (<a
 * href="https://github.com/wala/ML/issues/514">wala/ML#514</a>).
 */
public abstract class RandomDistribution extends TensorTypeAllocator {

  public RandomDistribution(PointsToSetVariable source) {
    super(source);
  }

  public RandomDistribution(CGNode node) {
    super(node);
  }

  // In the shape-first random signature `(shape, ..., dtype, seed, name)` the dtype argument sits
  // at position 3, not the allocator default of 1.
  @Override
  protected int getDTypeParameterPosition() {
    return 3;
  }
}
