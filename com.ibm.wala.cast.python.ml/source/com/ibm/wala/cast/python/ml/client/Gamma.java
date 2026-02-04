package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * A representation of the `tf.random.gamma` API in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/random/gamma">tf.random.gamma</a>
 *     API.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Gamma extends Ones {

  protected enum Parameters {
    SHAPE,
    ALPHA,
    BETA,
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

  public Gamma(PointsToSetVariable source) {
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

  protected int getAlphaParameterPosition() {
    return Parameters.ALPHA.getIndex();
  }

  protected String getAlphaParameterName() {
    return Parameters.ALPHA.getName();
  }

  protected int getBetaParameterPosition() {
    return Parameters.BETA.getIndex();
  }

  protected String getBetaParameterName() {
    return Parameters.BETA.getName();
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    Set<List<Dimension<?>>> shapes = super.getShapes(builder);

    if (shapes.isEmpty())
      throw new IllegalStateException("Cannot determine shape for mandatory shape parameter.");

    // Get the shape of the alpha parameter.
    OrdinalSet<InstanceKey> alphaPointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getAlphaParameterPosition(), this.getAlphaParameterName());
    Set<List<Dimension<?>>> alphaShapes = this.getShapesOfValue(builder, alphaPointsToSet);

    if (alphaShapes.isEmpty())
      throw new IllegalArgumentException("Cannot determine shape for mandatory alpha parameter.");

    OrdinalSet<InstanceKey> betaPointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getBetaParameterPosition(), this.getBetaParameterName());

    // If there is no beta parameter.
    if (betaPointsToSet == null || betaPointsToSet.isEmpty())
      // return shape `tf.concat([shape, tf.shape(alpha)], axis=0)`.
      shapes.forEach(
          shape -> {
            alphaShapes.forEach(
                aShape -> {
                  List<Dimension<?>> newShape = new ArrayList<>(shape);
                  newShape.addAll(aShape);
                  ret.add(newShape);
                });
          });
    else { // There is a beta parameter.
      // return shape `tf.concat([shape, tf.shape(alpha + beta)], axis=0)`.
      Set<List<Dimension<?>>> betaShapes = this.getShapesOfValue(builder, betaPointsToSet);

      shapes.forEach(
          shape -> {
            alphaShapes.forEach(
                aShape -> {
                  betaShapes.forEach(
                      bShape -> {
                        List<Dimension<?>> newShape = new ArrayList<>(shape);

                        // Right-aligned broadcasting
                        int aLen = aShape.size();
                        int bLen = bShape.size();
                        int maxLen = Math.max(aLen, bLen);

                        List<Dimension<?>> broadcastShape = new ArrayList<>();
                        for (int i = 0; i < maxLen; i++) {
                          // Get dimensions from right to left
                          Dimension<?> aDim = (i < aLen) ? aShape.get(aLen - 1 - i) : null;
                          Dimension<?> bDim = (i < bLen) ? bShape.get(bLen - 1 - i) : null;

                          if (aDim == null) broadcastShape.add(0, bDim);
                          else if (bDim == null) broadcastShape.add(0, aDim);
                          else broadcastShape.add(0, Dimension.max(aDim, bDim));
                        }

                        newShape.addAll(broadcastShape);
                        ret.add(newShape);
                      });
                });
          });
    }

    return ret;
  }
}
