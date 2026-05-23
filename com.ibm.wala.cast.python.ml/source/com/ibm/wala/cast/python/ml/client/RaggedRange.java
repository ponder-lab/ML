package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
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
import java.util.Locale;
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

  protected enum Parameters {
    STARTS,
    LIMITS,
    DELTAS,
    DTYPE,
    NAME,
    ROW_SPLITS_DTYPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
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

    for (Integer numPosArgs : this.getNumberOfPossiblePositionalArguments(builder)) {
      OrdinalSet<InstanceKey> startPts = OrdinalSet.empty();
      OrdinalSet<InstanceKey> limitPts = OrdinalSet.empty();
      OrdinalSet<InstanceKey> deltaPts = OrdinalSet.empty();

      if (numPosArgs == 0) {
        // Keyword only — fetched below from the keyword-fallback block.
      } else if (numPosArgs == 1) {
        // range(limits) or range(starts, limits=X)
        if (!this.isKeywordArgumentPresent(builder, Parameters.LIMITS.getName())) {
          limitPts =
              this.getArgumentPointsToSet(
                  builder, Parameters.STARTS.getIndex(), Parameters.LIMITS.getName());
        } else {
          startPts =
              this.getArgumentPointsToSet(
                  builder, Parameters.STARTS.getIndex(), Parameters.STARTS.getName());
          limitPts =
              this.getArgumentPointsToSet(
                  builder, UNDEFINED_PARAMETER_POSITION, Parameters.LIMITS.getName());
        }
      } else if (numPosArgs == 2) {
        // range(starts, limits)
        startPts =
            this.getArgumentPointsToSet(
                builder, Parameters.STARTS.getIndex(), Parameters.STARTS.getName());
        limitPts =
            this.getArgumentPointsToSet(
                builder, Parameters.LIMITS.getIndex(), Parameters.LIMITS.getName());
      } else if (numPosArgs >= 3) {
        // range(starts, limits, deltas)
        startPts =
            this.getArgumentPointsToSet(
                builder, Parameters.STARTS.getIndex(), Parameters.STARTS.getName());
        limitPts =
            this.getArgumentPointsToSet(
                builder, Parameters.LIMITS.getIndex(), Parameters.LIMITS.getName());
        deltaPts =
            this.getArgumentPointsToSet(
                builder, Parameters.DELTAS.getIndex(), Parameters.DELTAS.getName());
      }

      // Retrieve keyword args if not already set by positional (and not empty from initialization)
      if (startPts.isEmpty())
        startPts =
            OrdinalSet.unify(
                startPts,
                this.getArgumentPointsToSet(
                    builder, UNDEFINED_PARAMETER_POSITION, Parameters.STARTS.getName()));
      if (limitPts.isEmpty())
        limitPts =
            OrdinalSet.unify(
                limitPts,
                this.getArgumentPointsToSet(
                    builder, UNDEFINED_PARAMETER_POSITION, Parameters.LIMITS.getName()));
      if (deltaPts.isEmpty())
        deltaPts =
            OrdinalSet.unify(
                deltaPts,
                this.getArgumentPointsToSet(
                    builder, UNDEFINED_PARAMETER_POSITION, Parameters.DELTAS.getName()));

      // Keyword-only: `tf.ragged.range(starts=5)` semantically means `limits=5`. Apply the swap
      // AFTER the keyword fallback above so re-fetching `starts` can't undo it.
      if (numPosArgs == 0 && limitPts.isEmpty() && !startPts.isEmpty()) {
        limitPts = startPts;
        startPts = OrdinalSet.empty();
      }

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
        shape.add(RaggedDim.INSTANCE);
        ret.add(shape);
      } else {
        // All scalars. tf.ragged.range with scalars returns a 2D RaggedTensor with shape
        // (1, ceil((limit - start) / delta)). When start/limit/delta are statically-known
        // numeric literals and the cross-product yields a single length, pin that length as
        // a NumericDim; otherwise fall back to RaggedDim. Fix for
        // <a href="https://github.com/wala/ML/issues/546">wala/ML#546</a>.
        List<Dimension<?>> shape = new ArrayList<>();
        shape.add(new NumericDim(1));
        Integer staticLength = computeStaticInnerLength(startPts, limitPts, deltaPts);
        shape.add(staticLength != null ? new NumericDim(staticLength) : RaggedDim.INSTANCE);
        ret.add(shape);
      }
    }
    return ret;
  }

  /**
   * Computes the static inner-dimension length for the scalar form of {@code tf.ragged.range} when
   * {@code start}/{@code limit}/{@code delta} all resolve to compile-time numeric literals. Returns
   * {@code null} when any arg is non-literal, when {@code delta} could be zero (invalid at
   * runtime), or when the cross-product of literal values yields multiple distinct lengths.
   *
   * @param startPts Points-to set of the {@code start} argument (empty defaults to {@code 0}).
   * @param limitPts Points-to set of the {@code limit} argument; required.
   * @param deltaPts Points-to set of the {@code delta} argument (empty defaults to {@code 1}).
   * @return The single statically-computable inner length, or {@code null} if not derivable.
   */
  private static Integer computeStaticInnerLength(
      OrdinalSet<InstanceKey> startPts,
      OrdinalSet<InstanceKey> limitPts,
      OrdinalSet<InstanceKey> deltaPts) {
    Set<Double> limits = getPossibleDoubleValues(limitPts);
    if (limits.isEmpty() || limits.contains(null)) return null;

    Set<Double> starts = getPossibleDoubleValues(startPts);
    if (starts.contains(null)) return null;
    if (starts.isEmpty()) starts.add(0.0);

    Set<Double> deltas = getPossibleDoubleValues(deltaPts);
    if (deltas.contains(null)) return null;
    if (deltas.isEmpty()) deltas.add(1.0);

    Set<Integer> lengths = HashSetFactory.make();
    for (Double s : starts)
      for (Double l : limits)
        for (Double d : deltas) {
          if (d == 0.0) return null; // invalid at runtime
          lengths.add((int) Math.max(0, Math.ceil((l - s) / d)));
        }

    return lengths.size() == 1 ? lengths.iterator().next() : null;
  }
}
