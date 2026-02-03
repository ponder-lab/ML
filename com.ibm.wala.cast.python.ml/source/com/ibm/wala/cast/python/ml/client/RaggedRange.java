package com.ibm.wala.cast.python.ml.client;

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

  private static final Logger LOGGER = Logger.getLogger(RaggedRange.class.getName());

  protected enum Parameters {
    STARTS,
    LIMITS,
    DELTAS,
    DTYPE,
    NAME,
    ROW_SPLITS_DTYPE;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public RaggedRange(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (Integer numPosArgs : getNumberOfPossiblePositionalArguments(builder)) {
      OrdinalSet<InstanceKey> startPts = OrdinalSet.empty();
      OrdinalSet<InstanceKey> limitPts = OrdinalSet.empty();
      OrdinalSet<InstanceKey> deltaPts = OrdinalSet.empty();

      if (numPosArgs == 0) {
        // Keyword only
        startPts = getArgumentPointsToSet(builder, -1, Parameters.STARTS.getName());
        limitPts = getArgumentPointsToSet(builder, -1, Parameters.LIMITS.getName());
        deltaPts = getArgumentPointsToSet(builder, -1, Parameters.DELTAS.getName());

        if (limitPts.isEmpty() && !startPts.isEmpty()) {
          // tf.ragged.range(starts=5) -> limits=5
          limitPts = startPts;
          startPts = OrdinalSet.empty();
        }
      } else if (numPosArgs == 1) {
        // range(limits) or range(starts, limits=X)
        if (!isKeywordArgumentPresent(builder, Parameters.LIMITS.getName())) {
          limitPts = getArgumentPointsToSet(builder, 0, null);
        } else {
          startPts = getArgumentPointsToSet(builder, 0, null);
          limitPts = getArgumentPointsToSet(builder, -1, Parameters.LIMITS.getName());
        }
      } else if (numPosArgs == 2) {
        // range(starts, limits)
        startPts = getArgumentPointsToSet(builder, 0, null);
        limitPts = getArgumentPointsToSet(builder, 1, null);
      } else if (numPosArgs >= 3) {
        // range(starts, limits, deltas)
        startPts = getArgumentPointsToSet(builder, 0, null);
        limitPts = getArgumentPointsToSet(builder, 1, null);
        deltaPts = getArgumentPointsToSet(builder, 2, null);
      }

      // Retrieve keyword args if not already set by positional (and not empty from initialization)
      if (startPts.isEmpty())
        startPts =
            OrdinalSet.unify(
                startPts, getArgumentPointsToSet(builder, -1, Parameters.STARTS.getName()));
      if (limitPts.isEmpty())
        limitPts =
            OrdinalSet.unify(
                limitPts, getArgumentPointsToSet(builder, -1, Parameters.LIMITS.getName()));
      if (deltaPts.isEmpty())
        deltaPts =
            OrdinalSet.unify(
                deltaPts, getArgumentPointsToSet(builder, -1, Parameters.DELTAS.getName()));

      // Check for vectors
      boolean hasVector = false;
      Integer vectorLength = null;

      List<OrdinalSet<InstanceKey>> allSets = new ArrayList<>();
      allSets.add(startPts);
      allSets.add(limitPts);
      allSets.add(deltaPts);

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
}
