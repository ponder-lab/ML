package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSOR_ARRAY_READS;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAGetInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for the tensors a {@code tf.TensorArray} gives back: {@code stack}, {@code read},
 * {@code gather} and {@code concat}. Each has the dtype the array was built with. Its shape depends
 * on the array's size and element shape, which are not read, so it is unknown.
 *
 * <p>The summary body of each method reads the array's {@code dtype} into a local of its own. That
 * local is read in the body's own frame, whether this generator is anchored at the call's result
 * or, for producer delegation, at the body itself.
 *
 * @see <a href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/TensorArray">
 *     tf.TensorArray</a>
 */
public class TensorArrayRead extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(TensorArrayRead.class.getName());

  public TensorArrayRead(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Manual (node-based) anchor, for producer delegation from the tensor a method's body allocates.
   *
   * @param node The method's summary body.
   */
  public TensorArrayRead(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (CGNode body : getBodies(builder)) {
      IR ir = body.getIR();
      if (ir == null) continue;
      for (SSAInstruction instruction : ir.getInstructions()) {
        if (!(instruction instanceof SSAGetInstruction get)
            || !get.getDeclaredField().getName().toString().equals("dtype")) continue;
        OrdinalSet<InstanceKey> pts =
            builder
                .getPointerAnalysis()
                .getPointsToSet(builder.getPointerKeyForLocal(body, get.getDef()));
        if (pts == null || pts.isEmpty()) {
          LOGGER.fine(() -> "TensorArrayRead: no dtype reaches " + body + ".");
          ret.add(DType.UNKNOWN);
          continue;
        }
        ret.addAll(getDTypesFromDTypeArgument(builder, pts));
      }
    }
    if (ret.isEmpty()) ret.add(DType.UNKNOWN);
    return ret;
  }

  /**
   * The summary bodies this generator reads: its own node when it is anchored at one, and otherwise
   * the bodies its call dispatches to.
   */
  private Set<CGNode> getBodies(PropagationCallGraphBuilder builder) {
    Set<CGNode> ret = new HashSet<>();
    CGNode node = getNode();
    if (TENSOR_ARRAY_READS.contains(node.getMethod().getDeclaringClass().getReference())) {
      ret.add(node);
      return ret;
    }
    SSAAbstractInvokeInstruction invoke = getInvokeInstruction();
    if (invoke == null) return ret;
    for (CGNode target : builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite()))
      if (TENSOR_ARRAY_READS.contains(target.getMethod().getDeclaringClass().getReference()))
        ret.add(target);
    return ret;
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
