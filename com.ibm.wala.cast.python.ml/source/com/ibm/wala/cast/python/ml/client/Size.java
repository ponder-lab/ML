package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.size}. Returns a 0-D scalar {@code int32} tensor whose value is the total
 * number of elements of the input — output shape is intrinsic; dtype defaults to {@code int32} and
 * can be overridden via the {@code out_type} argument to {@code int64} (the {@link DType} lattice
 * models {@code int32}, {@code int64}, and a handful of others — {@code uint32}/{@code uint64} from
 * the runtime API aren't in the lattice). The current generator hardcodes {@code int32}; honoring
 * {@code out_type} is a clean follow-up if a fixture surfaces the need. See wala/ML#449 (Tier 4 —
 * fixed output).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/size">tf.size</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Size extends TensorGenerator {

  public Size(PointsToSetVariable source) {
    super(source);
  }

  public Size(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.add(Collections.emptyList());
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return Set.of(DType.INT32);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
