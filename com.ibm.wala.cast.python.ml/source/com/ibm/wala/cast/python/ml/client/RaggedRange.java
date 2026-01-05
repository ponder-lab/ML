package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE;
import static com.ibm.wala.cast.python.util.Util.getFunction;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of the `tf.ragged.range` operation.
 *
 * <p>Returns a `RaggedTensor` containing `range(starts, limits, deltas)`.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/ragged/range">tf.ragged.range</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class RaggedRange extends Range {

  @SuppressWarnings("unused")
  private static final Logger LOGGER = Logger.getLogger(RaggedRange.class.getName());

  public RaggedRange(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (Integer numArgs : getNumberOfPossiblePositionalArguments(builder)) {
      int startArgIdx = -1;
      int limitArgIdx = -1;
      int deltaArgIdx = -1;

      if (numArgs == 1) {
        // tf.ragged.range(limit) -> starts=0, limits=arg0, deltas=1
        limitArgIdx = 0;
      } else if (numArgs == 2) {
        // tf.ragged.range(start, limit) -> starts=arg0, limits=arg1, deltas=1
        startArgIdx = 0;
        limitArgIdx = 1;
      } else if (numArgs >= 3) {
        // tf.ragged.range(start, limit, delta)
        startArgIdx = 0;
        limitArgIdx = 1;
        deltaArgIdx = 2;
      } else {
        continue;
      }

      OrdinalSet<InstanceKey> startPTS = getPointsToSet(builder, startArgIdx);
      OrdinalSet<InstanceKey> limitPTS = getPointsToSet(builder, limitArgIdx);
      OrdinalSet<InstanceKey> deltaPTS = getPointsToSet(builder, deltaArgIdx);

      // Check for vectors
      boolean hasVector = false;
      Integer vectorLength = null;

      List<OrdinalSet<InstanceKey>> allSets = new ArrayList<>();
      if (startArgIdx != -1) allSets.add(startPTS);
      if (limitArgIdx != -1) allSets.add(limitPTS);
      if (deltaArgIdx != -1) allSets.add(deltaPTS);

      for (OrdinalSet<InstanceKey> pts : allSets) {
        if (pts != null) {
          for (InstanceKey ik : pts) {
            if (ik instanceof AllocationSiteInNode) {
              AllocationSiteInNode asin = (AllocationSiteInNode) ik;
              TypeReference ref = asin.getConcreteType().getReference();
              if (ref.equals(PythonTypes.list) || ref.equals(PythonTypes.tuple)) {
                hasVector = true;
                OrdinalSet<InstanceKey> catalog =
                    pointerAnalysis.getPointsToSet(
                        ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                            .getPointerKeyForObjectCatalog(asin));
                vectorLength = catalog.size();
              }
            }
          }
        }
      }

      if (hasVector) {
        // Return 2D ragged tensor shape
        List<Dimension<?>> shape = new ArrayList<>();
        shape.add(new NumericDim(vectorLength != null ? vectorLength : -1));
        shape.add(null); // Ragged dimension
        ret.add(shape);
      } else {
        // All scalars.
        // tf.ragged.range with scalars returns a 2D RaggedTensor with shape (1, None).
        // e.g. tf.ragged.range(3, 18, 3) -> [[3, 6, 9, 12, 15]]
        List<Dimension<?>> shape = new ArrayList<>();
        shape.add(new NumericDim(1));
        shape.add(null); // Ragged dimension
        ret.add(shape);
      }
    }
    return ret;
  }

  private OrdinalSet<InstanceKey> getPointsToSet(PropagationCallGraphBuilder builder, int argIdx) {
    if (argIdx == -1) return null;
    int vn =
        this.getNode().getMethod().isStatic()
            ? this.getNode().getIR().getParameter(argIdx)
            : this.getNode().getIR().getParameter(argIdx + 1);
    return builder
        .getPointerAnalysis()
        .getPointsToSet(
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(this.getNode(), vn));
  }

  @Override
  protected String getSignature() {
    TypeReference function = getFunction(this.getSource());
    return TYPE_REFERENCE_TO_SIGNATURE.get(function);
  }
}
