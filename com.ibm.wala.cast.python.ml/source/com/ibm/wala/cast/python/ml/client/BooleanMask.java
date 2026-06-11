package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.boolean_mask}. Output dtype is inherited from the {@code tensor} input.
 * Output shape collapses the {@code mask.rank} axes starting at {@code axis} (i.e. the half-open
 * range {@code [axis, axis + mask.rank)}) into a single dimension whose length is the number of
 * {@code True} entries in {@code mask} — a runtime quantity static analysis cannot recover —
 * keeping the rest of {@code tensor.shape}. That collapsed dimension is therefore emitted as a
 * {@link DynamicDim}; the rank and the surrounding dimensions are precise. See wala/ML#449 (Tier
 * 8).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/boolean_mask">tf.boolean_mask</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class BooleanMask extends PassThroughUnaryTensorGenerator {

  public BooleanMask(PointsToSetVariable source) {
    super(source);
  }

  public BooleanMask(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "tensor";
  }

  /**
   * Derives the output shape as {@code tensor.shape[:axis] + [Dynamic] + tensor.shape[axis +
   * mask.rank:]}: {@code boolean_mask} replaces the {@code mask.rank} axes starting at {@code axis}
   * with a single dimension whose extent is the runtime count of {@code True} entries (emitted as a
   * {@link DynamicDim}). {@code axis} defaults to {@code 0} and a constant override is honored.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes (with a dynamic masked dimension), or {@code null}
   *     (⊤) when either input's shape is unknown or {@code axis} is a non-constant.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // tensor is arg 0; mask is arg 1; axis is arg 2 (default 0).
    Set<List<Dimension<?>>> tensorShapes =
        this.getShapes(builder, this.getArgumentValueNumber(builder, 0, "tensor", false));
    Set<List<Dimension<?>>> maskShapes =
        this.getShapes(builder, this.getArgumentValueNumber(builder, 1, "mask", false));
    if (tensorShapes == null || maskShapes == null) return null;

    // axis defaults to 0; collect the constant override value(s) across contexts (None -> 0).
    Set<Integer> axes = new HashSet<>();
    OrdinalSet<InstanceKey> axisPts = this.getArgumentPointsToSet(builder, 2, "axis");
    if (axisPts == null || axisPts.isEmpty()) {
      axes.add(0);
    } else {
      for (InstanceKey ik : axisPts) {
        if (!(ik instanceof ConstantKey)) return null;
        Object val = ((ConstantKey<?>) ik).getValue();
        if (val == null) axes.add(0); // None -> default axis 0
        else if (val instanceof Number) axes.add(((Number) val).intValue());
        else return null;
      }
    }

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (int axis : axes) {
      for (List<Dimension<?>> tensor : tensorShapes) {
        for (List<Dimension<?>> mask : maskShapes) {
          int maskRank = mask.size();
          if (axis < 0 || maskRank < 1 || axis + maskRank > tensor.size()) continue;
          List<Dimension<?>> out = new ArrayList<>(tensor.subList(0, axis));
          out.add(DynamicDim.INSTANCE);
          out.addAll(tensor.subList(axis + maskRank, tensor.size()));
          ret.add(out);
        }
      }
    }
    return ret.isEmpty() ? null : ret;
  }
}
