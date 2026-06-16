package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.gather_nd}. Output dtype is inherited from the {@code params} input.
 * Output shape is {@code indices.shape[:-1] + params.shape[indices.shape[-1]:]}: the leading {@code
 * indices.shape[:-1]} indexes the gather and each innermost index of depth {@code
 * indices.shape[-1]} selects a {@code params.shape[indices.shape[-1]:]} slice. Derivable when the
 * index depth (the last dimension of {@code indices}) is a known constant. See wala/ML#449 (Tier
 * 8).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/gather_nd">tf.gather_nd</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class GatherNd extends PassThroughUnaryTensorGenerator {

  public GatherNd(PointsToSetVariable source) {
    super(source);
  }

  public GatherNd(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "params";
  }

  /**
   * Derives the output shape as {@code indices.shape[:-1] + params.shape[k:]}, where {@code k =
   * indices.shape[-1]} is the index depth. Each innermost index of length {@code k} selects a
   * {@code params.shape[k:]} slice, and those slices are arranged according to {@code
   * indices.shape[:-1]}. Requires the index depth to be a known constant.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when either input's shape is
   *     unknown, {@code indices} has rank 0, or the index depth is not a constant.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // params is arg 0; indices is arg 1.
    Set<List<Dimension<?>>> paramsShapes =
        this.getShapes(builder, this.getArgumentValueNumber(builder, 0, "params", false));
    Set<List<Dimension<?>>> indicesShapes =
        this.getShapes(builder, this.getArgumentValueNumber(builder, 1, "indices", false));
    if (paramsShapes == null || indicesShapes == null) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> indices : indicesShapes) {
      // indices needs a trailing axis whose (constant) extent is the index depth.
      if (indices.isEmpty()) continue;
      Dimension<?> depthDim = indices.get(indices.size() - 1);
      if (!(depthDim instanceof NumericDim)) continue;
      int depth = ((NumericDim) depthDim).value();
      for (List<Dimension<?>> params : paramsShapes) {
        // The index depth can address at most the full rank of params.
        if (depth > params.size()) continue;
        List<Dimension<?>> out = new ArrayList<>(indices.subList(0, indices.size() - 1));
        out.addAll(params.subList(depth, params.size()));
        ret.add(out);
      }
    }
    return ret.isEmpty() ? null : ret;
  }
}
