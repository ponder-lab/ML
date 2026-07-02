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
      // Track which args the call explicitly provides—distinct from "PTS happens to be empty".
      // `getArgumentPointsToSet` can return `OrdinalSet.empty()` for arg-present-but-unresolvable
      // cases too; treating "empty" as "omitted" would make the inferred length unsound.
      boolean startProvided = false;
      boolean limitProvided = false;
      boolean deltaProvided = false;

      if (numPosArgs == 0) {
        // Keyword only—fetched below from the keyword-fallback block.
      } else if (numPosArgs == 1) {
        // range(limits) or range(starts, limits=X)
        if (!this.isKeywordArgumentPresent(builder, Parameters.LIMITS.getName())) {
          limitPts =
              this.getArgumentPointsToSet(
                  builder, Parameters.STARTS.getIndex(), Parameters.LIMITS.getName());
          limitProvided = true;
        } else {
          startPts =
              this.getArgumentPointsToSet(
                  builder, Parameters.STARTS.getIndex(), Parameters.STARTS.getName());
          startProvided = true;
          limitPts =
              this.getArgumentPointsToSet(
                  builder, UNDEFINED_PARAMETER_POSITION, Parameters.LIMITS.getName());
          limitProvided = true;
        }
      } else if (numPosArgs == 2) {
        // range(starts, limits)
        startPts =
            this.getArgumentPointsToSet(
                builder, Parameters.STARTS.getIndex(), Parameters.STARTS.getName());
        startProvided = true;
        limitPts =
            this.getArgumentPointsToSet(
                builder, Parameters.LIMITS.getIndex(), Parameters.LIMITS.getName());
        limitProvided = true;
      } else if (numPosArgs >= 3) {
        // range(starts, limits, deltas)
        startPts =
            this.getArgumentPointsToSet(
                builder, Parameters.STARTS.getIndex(), Parameters.STARTS.getName());
        startProvided = true;
        limitPts =
            this.getArgumentPointsToSet(
                builder, Parameters.LIMITS.getIndex(), Parameters.LIMITS.getName());
        limitProvided = true;
        deltaPts =
            this.getArgumentPointsToSet(
                builder, Parameters.DELTAS.getIndex(), Parameters.DELTAS.getName());
        deltaProvided = true;
      }

      // Retrieve keyword args if not already set by positional. Use `isKeywordArgumentPresent` to
      // know whether the arg is actually in the call, not just whether the PTS resolved.
      if (!startProvided && this.isKeywordArgumentPresent(builder, Parameters.STARTS.getName())) {
        startPts =
            this.getArgumentPointsToSet(
                builder, UNDEFINED_PARAMETER_POSITION, Parameters.STARTS.getName());
        startProvided = true;
      }
      if (!limitProvided && this.isKeywordArgumentPresent(builder, Parameters.LIMITS.getName())) {
        limitPts =
            this.getArgumentPointsToSet(
                builder, UNDEFINED_PARAMETER_POSITION, Parameters.LIMITS.getName());
        limitProvided = true;
      }
      if (!deltaProvided && this.isKeywordArgumentPresent(builder, Parameters.DELTAS.getName())) {
        deltaPts =
            this.getArgumentPointsToSet(
                builder, UNDEFINED_PARAMETER_POSITION, Parameters.DELTAS.getName());
        deltaProvided = true;
      }

      // Keyword-only: `tf.ragged.range(starts=5)` semantically means `limits=5`. Apply the swap
      // AFTER the keyword fallback above so re-fetching `starts` can't undo it.
      if (numPosArgs == 0 && !limitProvided && startProvided) {
        limitPts = startPts;
        limitProvided = true;
        startPts = OrdinalSet.empty();
        startProvided = false;
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
              TypeReference ref = asin.concreteType().getReference();
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
        Integer staticLength =
            computeStaticInnerLength(
                startPts, startProvided, limitPts, limitProvided, deltaPts, deltaProvided);
        shape.add(staticLength != null ? new NumericDim(staticLength) : RaggedDim.INSTANCE);
        ret.add(shape);
      }
    }
    return ret;
  }

  /**
   * Computes the static inner-dimension length for the scalar form of {@code tf.ragged.range} when
   * {@code start}/{@code limit}/{@code delta} all resolve to compile-time numeric literals. Returns
   * {@code null} when {@code limit} is omitted, when any provided arg is non-literal or
   * unresolvable, when {@code delta} could be zero (invalid at runtime), or when the cross-product
   * of literal values yields multiple distinct lengths. {@code start}/{@code delta} default to
   * {@code 0}/{@code 1} only when their {@code *Provided} flag is false (the runtime default for an
   * omitted arg)—empty PTS for a *provided* arg means "unresolvable" and forces a fallback to
   * {@code RaggedDim}.
   *
   * @param startPts Points-to set of the {@code start} argument; consulted only when {@code
   *     startProvided}.
   * @param startProvided Whether the call provided an explicit {@code start} (positional or
   *     keyword).
   * @param limitPts Points-to set of the {@code limit} argument; required (mandatory at runtime).
   * @param limitProvided Whether the call provided an explicit {@code limit}.
   * @param deltaPts Points-to set of the {@code delta} argument; consulted only when {@code
   *     deltaProvided}.
   * @param deltaProvided Whether the call provided an explicit {@code delta}.
   * @return The single statically-computable inner length, or {@code null} if not derivable.
   */
  private static Integer computeStaticInnerLength(
      OrdinalSet<InstanceKey> startPts,
      boolean startProvided,
      OrdinalSet<InstanceKey> limitPts,
      boolean limitProvided,
      OrdinalSet<InstanceKey> deltaPts,
      boolean deltaProvided) {
    if (!limitProvided) return null;

    // `getPossibleDoubleValues` returns null on a non-constant PTS key (not statically
    // resolvable, wala/ML#669) and throws `IllegalStateException` on a non-numeric constant;
    // degrade both to RaggedDim instead of crashing the analysis. Mirrors the established
    // "modeling gap → soft fallback" pattern in e.g. `Reshape.getShapes`.
    Set<Double> limits;
    Set<Double> starts;
    Set<Double> deltas;
    try {
      limits = getPossibleDoubleValues(limitPts);
      starts = startProvided ? getPossibleDoubleValues(startPts) : new java.util.HashSet<>();
      deltas = deltaProvided ? getPossibleDoubleValues(deltaPts) : new java.util.HashSet<>();
    } catch (IllegalStateException e) {
      return null;
    }

    if (limits == null || starts == null || deltas == null) return null;

    if (limits.isEmpty() || limits.contains(null)) return null;
    if (startProvided && (starts.isEmpty() || starts.contains(null))) return null;
    if (deltaProvided && (deltas.isEmpty() || deltas.contains(null))) return null;

    if (!startProvided) starts.add(0.0);
    if (!deltaProvided) deltas.add(1.0);

    Set<Integer> lengths = HashSetFactory.make();
    for (Double s : starts)
      for (Double l : limits)
        for (Double d : deltas) {
          if (d == 0.0) return null; // invalid at runtime
          // NaN/Infinity would silently degrade through `(int) Math.ceil(NaN) == 0`, pinning a
          // bogus `NumericDim(0)` instead of falling back to `RaggedDim`. Force the fallback.
          if (!Double.isFinite(s) || !Double.isFinite(l) || !Double.isFinite(d)) return null;
          lengths.add((int) Math.max(0, Math.ceil((l - s) / d)));
          // Short-circuit: the caller only needs to know whether there's exactly one distinct
          // length; finishing the cross-product once we've seen two is wasted work.
          if (lengths.size() > 1) return null;
        }

    return lengths.size() == 1 ? lengths.iterator().next() : null;
  }
}
