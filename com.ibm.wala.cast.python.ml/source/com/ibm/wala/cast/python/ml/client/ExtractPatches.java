package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.image.extract_patches}. Output dtype is inherited from the {@code images}
 * input. Output shape is left at ⊤ for now: the precise shape depends on the {@code sizes}, {@code
 * strides}, {@code rates}, and {@code padding} arguments together with {@code images.shape}, which
 * is non-trivial to compute and not yet supported by the tier framework. See wala/ML#449 (Tier 8).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/image/extract_patches">tf.image.extract_patches</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ExtractPatches extends PassThroughUnaryTensorGenerator {

  public ExtractPatches(PointsToSetVariable source) {
    super(source);
  }

  public ExtractPatches(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "images";
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }
}
