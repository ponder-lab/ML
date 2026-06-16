package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
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
 * Generator for {@code tf.linalg.tensordot}. Output dtype is inherited from the {@code a} input.
 * Output shape is derived for the scalar {@code axes} form (contract the last {@code axes}
 * dimensions of {@code a} with the first {@code axes} dimensions of {@code b}, giving {@code
 * a.shape[:-axes] + b.shape[axes:]}); the 1-D and 2-D list forms of {@code axes} (contracting
 * explicitly enumerated axes) are left at ⊤. See wala/ML#449.
 *
 * <p>Extends {@link PassThroughUnaryTensorGenerator} for the dtype-from-input path only — the
 * shape-passthrough behavior is overridden to ⊤. The base class documents itself as "shape and
 * dtype passthrough"; using it here for *just* the dtype path is an intentional partial reuse,
 * acknowledged because extracting a separate dtype-only base would split a small piece of code
 * across two classes for a single-method gain. If more dtype-only-passthrough ops accumulate, a
 * shared {@code DTypePassThroughUnaryTensorGenerator} base is the natural refactor — tracked as
 * future work.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/linalg/tensordot">tf.linalg.tensordot</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Tensordot extends PassThroughUnaryTensorGenerator {

  public Tensordot(PointsToSetVariable source) {
    super(source);
  }

  public Tensordot(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "a";
  }

  /**
   * Derives the output shape for the scalar {@code axes} form: {@code tensordot(a, b, axes=n)}
   * contracts the last {@code n} dimensions of {@code a} with the first {@code n} dimensions of
   * {@code b}, so the output is {@code a.shape[:-n] + b.shape[n:]}. Returns ⊤ when {@code axes} is
   * not a constant integer (the 1-D and 2-D list forms enumerate specific axes and are not handled
   * here) or when either input's shape is unknown.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when the shape cannot be
   *     derived.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // axes is arg 2; only the scalar (integer) form is handled.
    OrdinalSet<InstanceKey> axesPts = this.getArgumentPointsToSet(builder, 2, "axes");
    if (axesPts == null || axesPts.isEmpty()) return null;
    Set<Integer> axesValues = new HashSet<>();
    for (InstanceKey ik : axesPts) {
      if (!(ik instanceof ConstantKey)) return null;
      Object val = ((ConstantKey<?>) ik).getValue();
      if (!(val instanceof Number)) return null;
      axesValues.add(((Number) val).intValue());
    }

    Set<List<Dimension<?>>> aShapes =
        this.getShapes(builder, this.getArgumentValueNumber(builder, 0, "a", false));
    Set<List<Dimension<?>>> bShapes =
        this.getShapes(builder, this.getArgumentValueNumber(builder, 1, "b", false));
    if (aShapes == null || bShapes == null) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (int axes : axesValues) {
      if (axes < 0) continue;
      for (List<Dimension<?>> a : aShapes) {
        if (axes > a.size()) continue;
        for (List<Dimension<?>> b : bShapes) {
          if (axes > b.size()) continue;
          List<Dimension<?>> out = new ArrayList<>(a.subList(0, a.size() - axes));
          out.addAll(b.subList(axes, b.size()));
          ret.add(out);
        }
      }
    }
    return ret.isEmpty() ? null : ret;
  }
}
