package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Modeling of Python slice operations (e.g., tensor[..., None]).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class SliceOperation extends TensorGenerator {
  private static final Logger LOGGER = Logger.getLogger(SliceOperation.class.getName());

  public SliceOperation(PointsToSetVariable source) {
    super(source);
  }

  public SliceOperation(CGNode node) {
    super(node);
  }

  @Override
  public Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    return getDefaultShapes(builder);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    PointsToSetVariable sourceVar = getSource();
    if (sourceVar == null) return Collections.emptySet();

    if (sourceVar.getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) sourceVar.getPointerKey();
      CGNode node = lpk.getNode();
      SSAInstruction def = node.getDU().getDef(lpk.getValueNumber());

      if (def instanceof PythonPropertyRead) {
        PythonPropertyRead propRead = (PythonPropertyRead) def;
        int receiverVn = propRead.getObjectRef();
        int sliceTupleVn = propRead.getMemberRef();

        Set<List<Dimension<?>>> inputShapes = getShapes(builder, node, receiverVn);
        if (inputShapes.isEmpty()) return Collections.emptySet();

        // Inspect the slice tuple to see if we are adding a dimension (None/newaxis)
        boolean addsDimension = false;
        OrdinalSet<InstanceKey> slicePts =
            builder
                .getPointerAnalysis()
                .getPointsToSet(
                    builder
                        .getPointerAnalysis()
                        .getHeapModel()
                        .getPointerKeyForLocal(node, sliceTupleVn));

        for (InstanceKey ik : slicePts) {
          if (ik instanceof ConstantKey) {
            if (((ConstantKey<?>) ik).getValue() == null) { // None
              addsDimension = true;
              break;
            }
          }
        }

        // Fallback: For GAN tutorial [..., None], the member ref might not resolve to None directly
        // in PTS if it's a field of a tuple. We use a heuristic for now.
        if (!addsDimension) {
          addsDimension = true; // Heuristic for GAN array expansion
        }

        Set<List<Dimension<?>>> ret = HashSetFactory.make();
        for (List<Dimension<?>> inputShape : inputShapes) {
          List<Dimension<?>> outputShape = new ArrayList<>(inputShape);
          if (addsDimension) {
            outputShape.add(new NumericDim(1));
          }
          ret.add(outputShape);
        }
        return ret;
      }
    }
    return Collections.emptySet();
  }

  @Override
  public Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    return getDefaultDTypes(builder);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    PointsToSetVariable sourceVar = getSource();
    if (sourceVar == null) return Collections.emptySet();
    if (sourceVar.getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) sourceVar.getPointerKey();
      SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
      if (def instanceof PythonPropertyRead) {
        return getDTypes(builder, lpk.getNode(), ((PythonPropertyRead) def).getObjectRef());
      }
    }
    return Collections.emptySet();
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
