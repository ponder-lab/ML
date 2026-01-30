package com.ibm.wala.cast.python.ml.client;

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
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Poisson(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SHAPE.getName();
  }

  protected int getLamParameterPosition() {
    return Parameters.LAM.getIndex();
  }

  protected String getLamParameterName() {
    return Parameters.LAM.getName();
  }

  protected int getLamParameterValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getLamParameterPosition(), getLamParameterName(), false);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // ...
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
