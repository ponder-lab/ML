package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Multiply.Parameters.X;
import static com.ibm.wala.cast.python.ml.client.Multiply.Parameters.Y;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.areBroadcastable;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.getBroadcastedShapes;
import static com.ibm.wala.cast.python.util.Util.getFunction;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of a multiply operation in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/multiply">tf.multiply</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Multiply extends ZerosLike {

  @SuppressWarnings("unused")
  private static final Logger logger = getLogger(Multiply.class.getName());

  protected enum Parameters {
    X,
    Y,
    NAME
  }

  /**
   * The dtype argument is not explicitly provided to multiply(); rather, the dtype is inferred from
   * the `x` argument.
   *
   * @see <a
   *     href="https://www.tensorflow.org/api_docs/python/tf/math/multiply#returns">tf.math.multiply
   *     - Returns</a>.
   */
  protected static final int DTYPE_PARAMETER_POSITION = -1;

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE_PARAMETER_POSITION;
  }

  public Multiply(PointsToSetVariable source) {
    super(source);
  }

  protected int getXParameterPosition() {
    return X.ordinal();
  }

  protected int getXArgumentValueNumber(PropagationCallGraphBuilder builder) {
    // TODO: Handle keyword arguments.
    return this.getArgumentValueNumber(builder, this.getXParameterPosition());
  }

  protected int getYParameterPosition() {
    return Y.ordinal();
  }

  protected int getYArgumentValueNumber(PropagationCallGraphBuilder builder) {
    // TODO: Handle keyword arguments.
    return this.getArgumentValueNumber(builder, this.getYParameterPosition());
  }

  /**
   * Returns the TensorFlow function signature represented by this generator.
   *
   * @return The TensorFlow function signature represented by this generator.
   */
  @Override
  protected String getSignature() {
    TypeReference function = getFunction(this.getSource());
    return TYPE_REFERENCE_TO_SIGNATURE.get(function);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The resulting shape is the broadcasted shape of the shapes of x and y.
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    Set<List<Dimension<?>>> xShapes =
        this.getShapes(builder, this.getXArgumentValueNumber(builder));
    Set<List<Dimension<?>>> yShapes =
        this.getShapes(builder, this.getYArgumentValueNumber(builder));

    for (List<Dimension<?>> xShape : xShapes)
      for (List<Dimension<?>> yShape : yShapes)
        if (areBroadcastable(xShape, yShape)) ret.add(getBroadcastedShapes(xShape, yShape));
        else throw new NonBroadcastableShapesException(this, xShape, yShape);

    return ret;
  }
}
