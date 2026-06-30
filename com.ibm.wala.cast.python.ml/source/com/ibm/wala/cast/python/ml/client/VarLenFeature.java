package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for the SparseTensor a {@code tf.io.VarLenFeature(dtype)} represents: when a
 * variable-length feature is parsed (e.g. by {@code tf.io.parse_single_example}) it yields a {@code
 * tf.sparse.SparseTensor} whose values carry the feature's dtype and whose dense shape is rank-1
 * with a dynamic length. Modeling it as that SparseTensor lets {@code parse_single_example} (a
 * pass-through of its feature dict) and a downstream {@code tf.sparse.to_dense} type the parsed
 * value (wala/ML#645).
 *
 * <p>{@code dtype} is the sole argument and there is no {@code shape} argument, so the base {@link
 * TensorTypeAllocator}'s dtype slot is read while the shape comes from the API contract (see {@link
 * #getDefaultShapes}).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/io/VarLenFeature">tf.io.VarLenFeature</a>.
 */
public class VarLenFeature extends TensorTypeAllocator {

  @Override
  protected boolean producesSparseTensor() {
    return true;
  }

  protected enum Parameters {
    DTYPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public VarLenFeature(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a manual-node generator for the SparseTensor allocated in {@code
   * VarLenFeature.do()}. Used by {@link TensorGenerator#createManualGenerator(CGNode,
   * PropagationCallGraphBuilder)} when that SparseTensor reaches a consumer (e.g. {@code
   * tf.sparse.to_dense}) through a container such as a feature dict, where the points-to walk lands
   * on the allocation site rather than the {@code tf.io.VarLenFeature} call. Reads the feature's
   * {@code dtype} from the {@code do()} method's parameter (wala/ML#646).
   *
   * @param node The {@code VarLenFeature.do()} call-graph node that allocated the SparseTensor.
   */
  public VarLenFeature(CGNode node) {
    super(node);
  }

  /**
   * Returns the rank-1 shape {@code (?,)} that the {@code VarLenFeature} API contract guarantees,
   * rather than ⊤. A variable-length feature parses to a sparse tensor with one axis whose length
   * varies per example, so the rank (1) is statically known even though the single dimension is
   * dynamic. This is a contract-model refinement, not a recovered argument: there is no {@code
   * shape} argument to read, but the API's documented behavior fixes the rank. {@link DynamicDim}
   * marks the dynamic axis, the same single-dynamic-axis idiom {@code tf.range} uses (wala/ML#647).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return A single rank-1 shape with a {@link DynamicDim} length.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    List<Dimension<?>> rank1 = new ArrayList<>();
    rank1.add(DynamicDim.INSTANCE);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.add(rank1);
    return ret;
  }

  /** No {@code shape} argument: the dense shape comes from the API contract, not an argument. */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /** The dtype is the sole argument. */
  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
