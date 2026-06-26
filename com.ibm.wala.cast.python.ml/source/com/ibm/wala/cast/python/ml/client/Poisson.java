package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A representation of the `tf.random.poisson` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/random/poisson">tf.random.poisson</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Poisson extends TensorTypeAllocator {

  protected enum Parameters {
    SHAPE,
    LAM,
    DTYPE,
    SEED,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Poisson(PointsToSetVariable source) {
    super(source);
  }

  public Poisson(CGNode node) {
    super(node);
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

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    Set<List<Dimension<?>>> shapes = super.getShapes(builder);

    // The shape argument is unresolvable (content-dependent) and the output rank rides on it, so
    // floor to ⊤ rather than aborting the whole analysis. super.getShapes already computes the
    // precise shape (and recovers a `.shape` argument, wala/ML#604) when it is resolvable.
    // wala/ML#611.
    if (shapes == null || shapes.isEmpty()) return null;

    // Get the shape of the lam parameter.
    OrdinalSet<InstanceKey> lamPTS =
        this.getArgumentPointsToSet(
            builder, this.getLamParameterPosition(), this.getLamParameterName());

    // lam's shape is part of the output rank; ⊤ if the mandatory argument is unresolvable.
    // wala/ML#611.
    if (lamPTS == null || lamPTS.isEmpty()) return null;

    Set<List<Dimension<?>>> lamShapes = this.getShapesOfValue(builder, lamPTS);

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
