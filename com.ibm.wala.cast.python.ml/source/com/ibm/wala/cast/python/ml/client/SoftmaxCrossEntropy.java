package com.ibm.wala.cast.python.ml.client;

import static java.util.Collections.emptySet;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code tf.nn.softmax_cross_entropy_with_logits} and {@code
 * tf.nn.sparse_softmax_cross_entropy_with_logits}.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits">tf.nn.softmax_cross_entropy_with_logits</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits">tf.nn.sparse_softmax_cross_entropy_with_logits</a>
 */
public class SoftmaxCrossEntropy extends TensorGenerator {

  public SoftmaxCrossEntropy(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Both versions return a tensor with one fewer dimension than the logits (the last dimension is
    // reduced).
    OrdinalSet<InstanceKey> logitsPts = this.getArgumentPointsToSet(builder, 1, "logits");
    if (logitsPts == null || logitsPts.isEmpty()) return emptySet();

    Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, logitsPts);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : inputShapes) {
      if (!shape.isEmpty()) {
        List<Dimension<?>> newShape = new ArrayList<>(shape);
        newShape.remove(newShape.size() - 1);
        ret.add(newShape);
      }
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return Set.of(DType.FLOAT32);
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
