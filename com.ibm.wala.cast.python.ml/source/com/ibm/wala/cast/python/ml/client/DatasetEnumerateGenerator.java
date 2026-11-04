package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.findDefinition;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;
import java.util.Set;

/**
 * A generator for tensors created by {@code tf.data.Dataset.enumerate}. The element is the tuple
 * {@code (index, element)}: an int64 scalar index at position 0 and the receiver's element at
 * position 1, so a path below {@code [1]} reads the receiver's element at the rest of the path
 * (wala/ML#1190).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetEnumerateGenerator extends DatasetGenerator {

  public DatasetEnumerateGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetEnumerateGenerator(CGNode node) {
    super(node);
  }

  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    Set<TensorType> types = HashSetFactory.make();

    // Add the index type (int64 scalar)
    types.add(new TensorType(DType.INT64.name().toLowerCase(Locale.ROOT), Collections.emptyList()));

    // Add the underlying dataset types
    TensorGenerator underlying = getUnderlyingGenerator(builder);
    if (underlying != null) {
      types.addAll(underlying.getTensorTypes(builder));
    }

    return types;
  }

  /** The position of the index in the element. */
  private static final int INDEX_POSITION = 0;

  /** The position of the receiver's element in the element. */
  private static final int ELEMENT_POSITION = 1;

  /** The index's shape, a scalar. */
  private static final List<Dimension<?>> INDEX_SHAPE = Collections.emptyList();

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    return true;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    if (index == INDEX_POSITION) return Set.of(INDEX_SHAPE);
    if (index == ELEMENT_POSITION) {
      TensorGenerator underlying = getUnderlyingGenerator(builder);
      return underlying == null ? null : underlying.getShapes(builder);
    }
    return Collections.emptySet();
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    if (index == INDEX_POSITION) return EnumSet.of(DType.INT64);
    if (index == ELEMENT_POSITION) {
      TensorGenerator underlying = getUnderlyingGenerator(builder);
      return underlying == null ? EnumSet.of(DType.UNKNOWN) : underlying.getDTypes(builder);
    }
    return Collections.emptySet();
  }

  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1)
      return path.get(0) instanceof Integer index
          && (index == INDEX_POSITION || index == ELEMENT_POSITION);
    return isElementPosition(path.get(0))
        && getUnderlyingGenerator(builder) instanceof TupleElementProvider tep
        && tep.resolvesPath(builder, path.subList(1, path.size()));
  }

  @Override
  public boolean isStructureAt(PropagationCallGraphBuilder builder, List<Object> path) {
    return !path.isEmpty()
        && isElementPosition(path.get(0))
        && getUnderlyingGenerator(builder) instanceof TupleElementProvider tep
        && tep.isStructureAt(builder, path.subList(1, path.size()));
  }

  @Override
  public OptionalInt declaredArity(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.isEmpty()) return OptionalInt.of(2);
    if (isElementPosition(path.get(0))
        && getUnderlyingGenerator(builder) instanceof TupleElementProvider tep)
      return tep.declaredArity(builder, path.subList(1, path.size()));
    return OptionalInt.empty();
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getShapesForIndex(builder, index);
    if (!isElementPosition(path.get(0))) return Collections.emptySet();
    TensorGenerator underlying = getUnderlyingGenerator(builder);
    if (underlying instanceof TupleElementProvider tep)
      return tep.getShapesForPath(builder, path.subList(1, path.size()));
    return null;
  }

  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getDTypesForIndex(builder, index);
    if (!isElementPosition(path.get(0))) return Collections.emptySet();
    TensorGenerator underlying = getUnderlyingGenerator(builder);
    if (underlying instanceof TupleElementProvider tep)
      return tep.getDTypesForPath(builder, path.subList(1, path.size()));
    return EnumSet.of(DType.UNKNOWN);
  }

  /**
   * Whether a selector is the position of the receiver's element.
   *
   * @param selector A path's selector.
   * @return {@code true} iff it is the integer {@value #ELEMENT_POSITION}.
   */
  private static boolean isElementPosition(Object selector) {
    return selector instanceof Integer index && index == ELEMENT_POSITION;
  }

  /**
   * Retrieves the generator for the dataset being enumerated.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The underlying dataset generator, or null if not found.
   */
  public TensorGenerator getUnderlyingGenerator(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);

    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      for (InstanceKey valueIK : receiverPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
        if (asin != null) {
          int vn = findDefinition(asin.getNode(), asin);
          if (vn > 0) {
            PointerKey pk =
                builder
                    .getPointerAnalysis()
                    .getHeapModel()
                    .getPointerKeyForLocal(asin.getNode(), vn);
            PointsToSetVariable var = null;
            if (!builder.getPropagationSystem().isImplicit(pk)) {
              var = builder.getPropagationSystem().findOrCreatePointsToSet(pk);
            }
            if (var != null) {
              return TensorGeneratorFactory.getGenerator(var, builder);
            } else {
              return createManualGenerator(asin.getNode(), builder);
            }
          }
        }
      }
    }
    return null;
  }
}
