package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Poisson.Parameters.DTYPE;
import static com.ibm.wala.cast.python.ml.client.Poisson.Parameters.LAM;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * A representation of the `tf.random.poisson` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/random/poisson">tf.random.poisson</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Poisson extends Ones {

  protected enum Parameters {
    SHAPE,
    LAM,
    DTYPE,
    SEED,
    NAME
  }

  public Poisson(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE.ordinal();
  }

  protected int getLamParameterPosition() {
    return LAM.ordinal();
  }

  protected int getLamParameterValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(builder, this.getLamParameterPosition());
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    Set<List<Dimension<?>>> shapes = super.getShapes(builder);

    if (shapes.isEmpty())
      throw new IllegalStateException(
          "Cannot determine shape for " + this.getSignature() + " call.");

    // Get the shape of the lam parameter.
    Set<List<Dimension<?>>> lamShapes =
        this.getShapes(builder, this.getLamParameterValueNumber(builder));

    // return shape `tf.concat([shape, tf.shape(lam)], axis=0)`.
    shapes.forEach(
        shape -> {
          lamShapes.forEach(
              lShape -> {
                List<Dimension<?>> newShape = new ArrayList<>(shape);
                newShape.addAll(lShape);
                ret.add(newShape);
              });
        });

    return ret;
  }
}
