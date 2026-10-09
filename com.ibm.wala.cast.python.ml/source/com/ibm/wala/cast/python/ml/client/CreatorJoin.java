package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.Layout;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;

/**
 * The join of the generators of a value's several creators. A return value several {@code return}s
 * feed, or a returned local the arms of a conditional define, is any of its creators at run time,
 * so its generator is the join of theirs (wala/ML#1009's rule for the value reads, here for the
 * factory's dispatch of a callee's return value). The tensor types are the union of the parts' own,
 * each part's shapes paired with its own dtypes, so no pair arises that no creator produces. Read
 * by one axis, the shapes union and the unknown remainders disjoin, and the dtypes union, as the
 * value reads' join does. A creator that is no tensor contributes nothing to any axis, so a value
 * whose creators are a scalar tensor and a Python integer is the tensor's type, and a value with no
 * tensor creator is ⊥ on both axes, as the pairing convention requires. The parts' readings are the
 * memoized ones every other read of them sees.
 *
 * <p>The legacy shape view is the collapse of the record view, as for every record-capable
 * generator. No type feed is declared: a feed replaces one operation's unresolved seed from its
 * operands, and the parts are several operations.
 *
 * <p>Outside an analysis the resolver is absent and a query runs once (wala/ML#753), so a part
 * whose read returns to this join through a value read's creator join would recurse without bound;
 * a re-entered join reads as unknown there, the sound answer where nothing iterates. Under the
 * resolver the memoized reads decide re-entry themselves and the guard does not run.
 */
public class CreatorJoin extends TensorGenerator {

  /** The joins being evaluated on this thread outside an analysis, by the value's key. */
  private static final ThreadLocal<Set<PointerKey>> JOINS_IN_PROGRESS =
      ThreadLocal.withInitial(HashSetFactory::make);

  private final List<TensorGenerator> parts;

  /**
   * Constructs the join of the given generators, anchored at the value they are the creators of.
   *
   * @param source The value's {@link PointsToSetVariable}.
   * @param parts The creators' generators, at least two.
   */
  CreatorJoin(PointsToSetVariable source, List<TensorGenerator> parts) {
    super(source);
    this.parts = List.copyOf(parts);
  }

  /**
   * The creators' generators.
   *
   * @return The parts, in the order the creators were reached.
   */
  public List<TensorGenerator> getParts() {
    return this.parts;
  }

  /**
   * Evaluates a read over the parts, answering a re-entry outside an analysis with the given
   * unknown.
   *
   * @param builder The propagation call graph builder.
   * @param read The read over the parts.
   * @param unknown The answer to a re-entered read outside an analysis.
   * @return The read's result, or {@code unknown} on such a re-entry.
   */
  private <T> T guarded(PropagationCallGraphBuilder builder, Supplier<T> read, T unknown) {
    if (WorklistTypeResolver.active(builder) != null) return read.get();
    PointerKey key = this.getSource().getPointerKey();
    Set<PointerKey> inProgress = JOINS_IN_PROGRESS.get();
    if (!inProgress.add(key)) return unknown;
    try {
      return read.get();
    } finally {
      inProgress.remove(key);
    }
  }

  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    return this.guarded(
        builder,
        () -> {
          Set<TensorType> joined = HashSetFactory.make();
          boolean unknownPart = false;
          for (TensorGenerator part : this.parts) {
            Set<TensorType> types = part.getTensorTypes(builder);
            if (types == null) unknownPart = true;
            else joined.addAll(types);
          }
          // A part of unknown shape and dtype is the unknown tensor: alone, the join is that;
          // beside
          // typed parts it stands as the unknown-marked member.
          if (unknownPart) {
            if (joined.isEmpty()) return null;
            joined.add(TensorType.of(DType.UNKNOWN, null, Layout.DENSE));
          }
          return joined;
        },
        null);
  }

  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return this.guarded(
        builder,
        () -> {
          ShapeResult joined = ShapeResult.bottom();
          for (TensorGenerator part : this.parts)
            joined = joined.union(memoizedShapeResult(builder, part));
          return joined;
        },
        ShapeResult.unknown());
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder).toLegacy();
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.guarded(
        builder,
        () -> {
          Set<DType> joined = EnumSet.noneOf(DType.class);
          for (TensorGenerator part : this.parts) {
            Set<DType> dtypes = memoizedDTypes(builder, part);
            // The legacy dtype convention reads `null` as ⊤.
            if (dtypes == null) joined.add(DType.UNKNOWN);
            else joined.addAll(dtypes);
          }
          return joined;
        },
        EnumSet.of(DType.UNKNOWN));
  }

  /**
   * No origin evidence of its own: the join produces nothing, and each part's producing library
   * reaches the value through the dataflow from the part's own seed, as a layer call's result's
   * does. Reading the parts here would also classify through a parameter to its callers' producers,
   * which the frame's own origin stands in for (wala/ML#726, wala/ML#979). As for a delegation
   * whose underlying never resolved, this is no evidence, not the TensorFlow default (wala/ML#730).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return No origins.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.noneOf(TensorOrigin.class);
  }

  /** The join has no shape parameter of its own; each part reads its own. */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /** The join has no dtype parameter of its own; each part reads its own. */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
