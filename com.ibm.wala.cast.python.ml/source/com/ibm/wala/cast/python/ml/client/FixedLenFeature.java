package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * A generator for the dense tensor a {@code tf.io.FixedLenFeature(shape, dtype)} represents: when a
 * fixed-length feature is parsed (e.g. by {@code tf.io.parse_single_example}) it yields a dense
 * tensor whose shape is the feature's declared {@code shape} and whose dtype is the feature's
 * {@code dtype}. Modeling it as that tensor lets {@code parse_single_example} (a pass-through of
 * its feature dict) type the parsed value, so a dataset whose {@code map_func} returns such values
 * (the gpt-2 / NLPGNN input-pipeline shape) types its elements (wala/ML#655).
 *
 * <p>Unlike {@link VarLenFeature} (whose parsed value is a {@code tf.sparse.SparseTensor} with a
 * contract shape and only a {@code dtype} argument), a fixed-length feature carries both an
 * explicit {@code shape} ({@code dims}, the first argument) and a {@code dtype} ({@code type}, the
 * second argument), and parses to a dense tensor.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/io/FixedLenFeature">tf.io.FixedLenFeature</a>.
 */
public class FixedLenFeature extends TensorTypeAllocator {

  protected enum Parameters {
    DIMS,
    TYPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public FixedLenFeature(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a manual-node generator for the dense tensor allocated in {@code
   * FixedLenFeature.do()}. Used by {@link TensorGenerator#createManualGenerator(CGNode,
   * PropagationCallGraphBuilder)} when that tensor reaches a consumer through a container such as a
   * feature dict, where the points-to walk lands on the allocation site rather than the {@code
   * tf.io.FixedLenFeature} call. Reads the feature's {@code dims} and {@code type} from the {@code
   * do()} method's parameters (wala/ML#655).
   *
   * @param node The {@code FixedLenFeature.do()} call-graph node that allocated the tensor.
   */
  public FixedLenFeature(CGNode node) {
    super(node);
  }

  /** The shape is the first argument ({@code dims}). */
  @Override
  protected int getShapeParameterPosition() {
    return Parameters.DIMS.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.DIMS.getName();
  }

  /** The dtype is the second argument ({@code type}). */
  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.TYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.TYPE.getName();
  }
}
