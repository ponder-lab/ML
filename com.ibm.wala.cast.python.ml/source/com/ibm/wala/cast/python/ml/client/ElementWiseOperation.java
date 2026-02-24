package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.areBroadcastable;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.getBroadcastedShapes;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
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
    if (this.source.getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) this.source.getPointerKey();
      SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
      if (def instanceof SSABinaryOpInstruction) {
        return def.getUse(0);
      }
    }
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
    if (this.source.getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) this.source.getPointerKey();
      SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
      if (def instanceof SSABinaryOpInstruction) {
        return def.getUse(1);
      }
    }
    return this.getArgumentValueNumber(
        builder, this.getYParameterPosition(), this.getYParameterName(), false);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The resulting shape is the broadcasted shape of the shapes of x and y.
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    int xVn = this.getXArgumentValueNumber(builder);
    logger.fine("ElementWiseOperation getDefaultShapes xVn: " + xVn);
    if (xVn <= 0) return ret;
    Set<List<Dimension<?>>> xShapes = this.getShapes(builder, xVn);
    logger.fine("ElementWiseOperation getDefaultShapes xShapes: " + xShapes);

    int yVn = this.getYArgumentValueNumber(builder);
    logger.fine("ElementWiseOperation getDefaultShapes yVn: " + yVn);
    if (yVn <= 0) return ret;
    Set<List<Dimension<?>>> yShapes = this.getShapes(builder, yVn);
    logger.fine("ElementWiseOperation getDefaultShapes yShapes: " + yShapes);

    for (List<Dimension<?>> xShape : xShapes)
      for (List<Dimension<?>> yShape : yShapes)
        if (areBroadcastable(xShape, yShape)) ret.add(getBroadcastedShapes(xShape, yShape));
        else throw new NonBroadcastableShapesException(this, xShape, yShape);

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int vn = this.getXArgumentValueNumber(builder);
    logger.fine("ElementWiseOperation getDefaultDTypes vn: " + vn);
    if (vn <= 0) return Collections.emptySet();

    Set<DType> dtypes = this.getDTypes(builder, vn);
    logger.fine("ElementWiseOperation getDefaultDTypes dtypes: " + dtypes);
    return dtypes;
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
