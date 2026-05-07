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
 * TensorFlow API; the optional {@code dtype} argument that can override it is exposed in {@code
 * tensorflow.xml} (see {@code sequence_mask}'s {@code paramNames}, which include {@code dtype}) but
 * is not yet honored by this generator, so it emits {@code BOOL} unconditionally — similar to
 * {@link Argmax}'s {@code INT64} default (whose {@code output_type} override, by contrast, isn't
 * yet surfaced in the XML at all). Output shape is left at ⊤: the precise shape is {@code
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
}
