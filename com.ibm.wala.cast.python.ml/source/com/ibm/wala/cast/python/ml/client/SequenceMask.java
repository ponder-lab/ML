package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.sequence_mask}. Output dtype defaults to {@link DType#BOOL} per the
 * TensorFlow API; the optional {@code dtype} argument that overrides it (e.g. {@code
 * dtype=tf.int32}) is exposed in {@code tensorflow.xml} (see {@code sequence_mask}'s {@code
 * paramNames}, which include {@code dtype}) and is honored here by surfacing the {@code dtype}
 * parameter to the canonical dtype-argument dispatch in {@link TensorGenerator#getDTypes}, falling
 * back to {@code BOOL} when it is absent. Output shape is left at ⊤: the precise shape is {@code
 * (*lengths.shape, maxlen)} which requires combining one input's shape with another input's runtime
 * value. See wala/ML#449 (Tier 8).
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

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
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
}
