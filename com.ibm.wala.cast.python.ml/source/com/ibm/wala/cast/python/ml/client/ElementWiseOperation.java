package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.areBroadcastable;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.getBroadcastedShapes;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.EnumSet;
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

    logger.fine(
        () ->
            "EWO.getDefaultShapes entered with source=" + this.source + ", node=" + this.getNode());

    int xVn = this.getXArgumentValueNumber(builder);
    logger.fine(() -> "EWO.getDefaultShapes xVn: " + xVn);
    if (xVn <= 0) return null;
    Set<List<Dimension<?>>> xShapes = this.getOperandShapes(builder, xVn);
    logger.fine(() -> "EWO.getDefaultShapes xShapes: " + xShapes);
    if (xShapes == null) return null;

    int yVn = this.getYArgumentValueNumber(builder);
    logger.fine(() -> "EWO.getDefaultShapes yVn: " + yVn);
    if (yVn <= 0) return null;
    Set<List<Dimension<?>>> yShapes = this.getOperandShapes(builder, yVn);
    logger.fine(() -> "EWO.getDefaultShapes yShapes: " + yShapes);
    if (yShapes == null) return null;

    for (List<Dimension<?>> xShape : xShapes)
      for (List<Dimension<?>> yShape : yShapes)
        if (areBroadcastable(xShape, yShape)) ret.add(getBroadcastedShapes(xShape, yShape));
        else throw new NonBroadcastableShapesException(this, xShape, yShape);

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int vn = this.getXArgumentValueNumber(builder);
    logger.fine(() -> "ElementWiseOperation getDefaultDTypes vn: " + vn);
    if (vn <= 0) return EnumSet.of(DType.UNKNOWN);

    Set<DType> dtypes = this.getOperandDTypes(builder, vn);
    logger.fine(() -> "ElementWiseOperation getDefaultDTypes dtypes: " + dtypes);
    return dtypes;
  }

  /**
   * Resolves the shapes of an operand value number, with a targeted bypass for nested binary ops.
   *
   * <p>The base {@link #getShapes(PropagationCallGraphBuilder, CGNode, int)} path dispatches back
   * through {@link TensorGeneratorFactory#getGenerator} and then refuses to recurse when the
   * resulting generator is of the same class as {@code this} (to avoid infinite recursion with
   * other generator kinds). For nested binop chains like {@code (x - k1) / k2}, that class-skip
   * short-circuits the recursion and we lose the shape.
   *
   * <p>If the operand's def is itself a {@link SSABinaryOpInstruction}, this method constructs a
   * child {@link ElementWiseOperation} for the operand and invokes its {@link
   * #getDefaultShapes(PropagationCallGraphBuilder)} directly, bypassing the class-skip. Otherwise
   * it falls through to the standard {@link #getShapes(PropagationCallGraphBuilder, int)} path so
   * non-binop operands (invokes, slices, etc.) follow the normal generator dispatch.
   */
  private Set<List<Dimension<?>>> getOperandShapes(PropagationCallGraphBuilder builder, int vn) {
    ElementWiseOperation nested = getNestedForBinop(builder, vn);
    if (nested != null) return nested.getDefaultShapes(builder);
    return this.getShapes(builder, vn);
  }

  /** Dtype counterpart of {@link #getOperandShapes(PropagationCallGraphBuilder, int)}. */
  private Set<DType> getOperandDTypes(PropagationCallGraphBuilder builder, int vn) {
    ElementWiseOperation nested = getNestedForBinop(builder, vn);
    if (nested != null) return nested.getDefaultDTypes(builder);
    return this.getDTypes(builder, vn);
  }

  /**
   * If the given value number is defined by a binary op in {@code this}'s CGNode, returns a new
   * {@code ElementWiseOperation} whose source is the points-to set variable for that value number;
   * otherwise {@code null}.
   */
  private ElementWiseOperation getNestedForBinop(PropagationCallGraphBuilder builder, int vn) {
    CGNode node = this.getNode();
    SSAInstruction def = node.getDU().getDef(vn);
    if (!(def instanceof SSABinaryOpInstruction)) return null;
    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
    if (builder.getPropagationSystem().isImplicit(pk)) return null;
    PointsToSetVariable nestedSource = builder.getPropagationSystem().findOrCreatePointsToSet(pk);
    return new ElementWiseOperation(nestedSource);
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
