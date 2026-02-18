package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.areBroadcastable;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.getBroadcastedShapes;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
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

    public String getName() {
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
    return Parameters.X.getName();
  }

  protected int getXArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getXParameterPosition(), this.getXParameterName(), false);
  }

  protected int getYParameterPosition() {
    return Parameters.Y.getIndex();
  }

  protected String getYParameterName() {
    return Parameters.Y.getName();
  }

  protected int getYArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getYParameterPosition(), this.getYParameterName(), false);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The resulting shape is the broadcasted shape of the shapes of x and y.
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    OrdinalSet<InstanceKey> xPts =
        this.getArgumentPointsToSet(builder, getXParameterPosition(), getXParameterName());
    if (xPts == null || xPts.isEmpty()) return ret;

    Set<List<Dimension<?>>> xShapes = this.getShapesOfValue(builder, xPts);

    OrdinalSet<InstanceKey> yPts =
        this.getArgumentPointsToSet(builder, getYParameterPosition(), getYParameterName());
    if (yPts == null || yPts.isEmpty()) return ret;

    Set<List<Dimension<?>>> yShapes = this.getShapesOfValue(builder, yPts);

    for (List<Dimension<?>> xShape : xShapes)
      for (List<Dimension<?>> yShape : yShapes)
        if (areBroadcastable(xShape, yShape)) ret.add(getBroadcastedShapes(xShape, yShape));
        else throw new NonBroadcastableShapesException(this, xShape, yShape);

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int vn = this.getXArgumentValueNumber(builder);
    if (vn <= 0) return Collections.emptySet();

    OrdinalSet<InstanceKey> pts =
        builder
            .getPointerAnalysis()
            .getPointsToSet(
                builder
                    .getPointerAnalysis()
                    .getHeapModel()
                    .getPointerKeyForLocal(this.getNode(), vn));

    if (pts == null || pts.isEmpty()) return Collections.emptySet();

    return this.getDTypesOfValue(builder, pts);
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
