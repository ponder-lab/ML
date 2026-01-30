package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.areBroadcastable;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.getBroadcastedShapes;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of an element-wise operation in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/multiply">tf.multiply</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ElementWiseOperation extends ZerosLike {

  @SuppressWarnings("unused")
  private static final Logger logger = getLogger(ElementWiseOperation.class.getName());

  protected enum Parameters {
    X,
    Y,
    NAME;

    public String getParameterName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public ElementWiseOperation(PointsToSetVariable source) {
    super(source);
  }

  protected int getXParameterPosition() {
    return Parameters.X.getIndex();
  }

  protected String getXParameterName() {
    return Parameters.X.getParameterName();
  }

  protected int getXArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getXParameterPosition(), getXParameterName(), false);
  }

  protected int getYParameterPosition() {
    return Parameters.Y.getIndex();
  }

  protected String getYParameterName() {
    return Parameters.Y.getParameterName();
  }

  protected int getYArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getYParameterPosition(), getYParameterName(), false);
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

  /** No explicit dtype argument. Dtype is inferred from 'x'. */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
