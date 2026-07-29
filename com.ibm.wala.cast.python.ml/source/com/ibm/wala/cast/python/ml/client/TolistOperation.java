package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code ndarray.tolist()}: the receiver's elements as a nested Python list, so both
 * dtype and shape recover from the receiver. The runtime list of Python scalars erases the NumPy
 * dtype, but the downstream {@code np.array(list)} rebase re-derives exactly the receiver's element
 * type, so preserving it through the {@code tolist_result} allocation is the faithful reading for
 * the reader chains this models. See <a
 * href="https://github.com/wala/ML/issues/796">wala/ML#796</a>.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TolistOperation extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(TolistOperation.class.getName());

  /**
   * Parameter positions and keyword names for the {@code numpy/ndarray/tolist} synthetic. Ordinals
   * match the position in {@code numpy.xml}'s {@code paramNames} after the implicit {@code self}
   * receiver.
   */
  protected enum Parameters {
    /** The receiver array whose elements are listed; its dtype and shape are preserved. */
    NDARRAY;

    /**
     * Lowercase keyword name used in argument-resolution helpers.
     *
     * @return The lowercased enum name (e.g. {@code "ndarray"}).
     */
    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    /**
     * Positional index of this parameter, excluding the implicit {@code self} receiver.
     *
     * @return The zero-based positional index.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  public TolistOperation(PointsToSetVariable source) {
    super(source);
  }

  public TolistOperation(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Source anchor: resolve the receiver's value number in the caller frame first, since the
    // multi-strategy walk there covers producers (slice results, lexical reads) whose points-to
    // sets alone do not resolve.
    int receiverVn = getReceiverVn();
    if (this.source != null && receiverVn > 0) {
      try {
        Set<List<Dimension<?>>> shapes = getShapesOrSSAChain(builder, getNode(), receiverVn);
        if (shapes != null && !shapes.isEmpty()) return shapes;
      } catch (IllegalArgumentException e) {
        // Fall through to the caller-aware points-to path.
      }
    }
    OrdinalSet<InstanceKey> receiverPts =
        getArgumentPointsToSet(
            builder, Parameters.NDARRAY.getIndex(), Parameters.NDARRAY.getName());
    Set<List<Dimension<?>>> preserved = getShapesOfValue(builder, receiverPts);
    // ⊤ (unknown shape) when the receiver does not resolve: the value is still list-shaped like
    // its receiver, never ⊥.
    return preserved == null || preserved.isEmpty() ? null : preserved;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // See getDefaultShapes on the anchor split.
    int receiverVn = getReceiverVn();
    if (this.source != null && receiverVn > 0) {
      try {
        Set<DType> dTypes = getDTypesOrSSAChain(builder, getNode(), receiverVn);
        LOGGER.fine(
            () -> "Resolved tolist receiver dtypes via the caller-frame chain: " + dTypes + ".");
        if (dTypes != null && !dTypes.isEmpty() && !dTypes.equals(EnumSet.of(DType.UNKNOWN)))
          return dTypes;
      } catch (IllegalArgumentException e) {
        // Fall through to the caller-aware points-to path.
      }
    }
    OrdinalSet<InstanceKey> receiverPts =
        getArgumentPointsToSet(
            builder, Parameters.NDARRAY.getIndex(), Parameters.NDARRAY.getName());
    Set<DType> preserved = getDTypesOfValue(builder, receiverPts);
    return preserved == null || preserved.isEmpty() ? EnumSet.of(DType.UNKNOWN) : preserved;
  }

  /**
   * Returns the receiver's value number in the anchoring frame for a source-based instance: the
   * {@code x} of {@code x.tolist()}, read from the {@link PythonPropertyRead} that def'd the
   * invoke's function object. Returns {@code -1} for a manual anchor or when the structure does not
   * match; callers then use the caller-aware points-to path.
   *
   * @return The receiver's value number, or {@code -1} when unavailable.
   */
  private int getReceiverVn() {
    com.ibm.wala.cast.python.ssa.PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int funcVn = call.getUse(0);
      com.ibm.wala.ssa.SSAInstruction funcDef = getNode().getDU().getDef(funcVn);
      if (funcDef instanceof com.ibm.wala.cast.python.ssa.PythonPropertyRead)
        return ((com.ibm.wala.cast.python.ssa.PythonPropertyRead) funcDef).getObjectRef();
    }
    return -1;
  }

  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
