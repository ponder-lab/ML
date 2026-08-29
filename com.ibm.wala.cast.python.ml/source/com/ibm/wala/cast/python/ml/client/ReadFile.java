package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code tf.io.read_file(filename)}: the file's entire contents as a rank-0 {@code
 * string} tensor, by the API contract. Both axes are fixed by the contract rather than by any
 * argument, so there is nothing to resolve and nothing to degrade: whatever the filename argument
 * is, the result is a scalar string.
 *
 * <p>Modeling it matters upstream of image decoding: an unmodeled {@code read_file} leaves its
 * result ⊤ on both axes, and the decode that consumes it then starts a whole image pipeline from
 * nothing (wala/ML#853).
 *
 * @see <a href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/io/read_file">
 *     tf.io.read_file</a>.
 */
public class ReadFile extends TensorTypeAllocator {

  public ReadFile(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a manual-node generator for the tensor allocated in {@code read_file.do()}, used by
   * {@link TensorGenerator#createManualGenerator(CGNode, PropagationCallGraphBuilder)} when the
   * result reaches a consumer through producer delegation.
   *
   * @param node The {@code read_file.do()} call-graph node.
   */
  public ReadFile(CGNode node) {
    super(node);
  }

  /**
   * Returns the scalar shape the API contract fixes: the whole file is one string value.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return A single rank-0 shape.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.add(Collections.emptyList());
    return ret;
  }

  /**
   * Returns the {@code string} dtype the API contract fixes.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link DType#STRING}, alone.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.STRING);
  }

  /** No shape argument: the shape is the contract's. */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /** No dtype argument: the dtype is the contract's. */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
