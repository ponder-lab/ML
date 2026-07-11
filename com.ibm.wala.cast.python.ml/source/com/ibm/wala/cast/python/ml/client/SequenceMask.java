package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.sequence_mask}. Output dtype defaults to {@link DType#BOOL} per the
 * TensorFlow API; the optional {@code dtype} argument that overrides it (e.g. {@code
 * dtype=tf.int32}) is exposed in {@code tensorflow.xml} (see {@code sequence_mask}'s {@code
 * paramNames}, which include {@code dtype}) and is honored here by surfacing the {@code dtype}
 * parameter to the canonical dtype-argument dispatch in {@link TensorGenerator#getDTypes}, falling
 * back to {@code BOOL} when it is absent. Output shape is {@code (*lengths.shape, maxlen)}, derived
 * when {@code maxlen} is a constant; if {@code maxlen} is omitted it defaults to a runtime {@code
 * max(lengths)}, so the shape is left at ⊤ in that case. See wala/ML#449 (Tier 8).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/sequence_mask">tf.sequence_mask</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class SequenceMask extends PassThroughUnaryTensorGenerator {

  public SequenceMask(PointsToSetVariable source) {
    super(source);
  }

  public SequenceMask(CGNode node) {
    super(node);
  }

  /**
   * Derives the output shape as {@code lengths.shape + [maxlen]}: the mask adds a trailing axis of
   * width {@code maxlen} to the {@code lengths} shape. Derivable only when {@code maxlen} is a
   * constant integer; when it is omitted (or non-constant) it defaults to a runtime {@code
   * max(lengths)}, so this returns ⊤.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when {@code maxlen} is not a
   *     constant integer or the {@code lengths} shape is unknown.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // maxlen is arg 1; resolve its constant value(s) (None/non-constant -> runtime max(lengths)).
    OrdinalSet<InstanceKey> maxlenPts = this.getArgumentPointsToSet(builder, 1, "maxlen");
    if (maxlenPts == null || maxlenPts.isEmpty()) return null;
    Set<Integer> maxlens = new HashSet<>();
    for (InstanceKey ik : maxlenPts) {
      if (!(ik instanceof ConstantKey)) return null;
      Object val = ((ConstantKey<?>) ik).getValue();
      if (!(val instanceof Number)) return null;
      maxlens.add(((Number) val).intValue());
    }

    // lengths is arg 0.
    Set<List<Dimension<?>>> lengthsShapes =
        this.getShapes(builder, this.getArgumentValueNumber(builder, 0, "lengths", false));
    if (lengthsShapes == null) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> lengths : lengthsShapes) {
      for (int maxlen : maxlens) {
        List<Dimension<?>> out = new ArrayList<>(lengths);
        out.add(new NumericDim(maxlen));
        ret.add(out);
      }
    }
    return ret.isEmpty() ? null : ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.BOOL);
  }

  /**
   * Positional index of the {@code dtype} override argument. {@code tf.sequence_mask(lengths,
   * maxlen, dtype, name)} places {@code dtype} third among the user-facing arguments.
   *
   * @return The zero-based positional index of the {@code dtype} argument.
   */
  @Override
  protected int getDTypeParameterPosition() {
    return 2;
  }

  /**
   * Keyword name of the {@code dtype} override argument.
   *
   * @return {@code "dtype"}.
   */
  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }

  /**
   * Collapse-safe record view (wala/ML#718): this generator transforms its input shapes in {@link
   * #getDefaultShapes}, which the pass-through identity record path would bypass, so the record
   * view routes through the legacy transform until a member-wise upgrade.
   *
   * @param builder The propagation call graph builder.
   * @return The transformed result, with any partial input collapsed by the legacy view.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }
}
