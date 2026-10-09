package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;

/**
 * A generator for the {@code embeddings} weight of a {@code tf.keras.layers.Embedding}: a float32
 * matrix of {@code input_dim} rows and {@code output_dim} columns, the two leading arguments of the
 * layer's constructor, which the constructor's summary passes to the weight's own allocating
 * function as plain arguments. The weight is what a subclass overriding {@code call} gathers from,
 * so this generator types {@code self.embeddings} where the {@code EmbeddingCall} rule, which types
 * the layer's own call, never applies. The rows-by-columns construction is {@link EyeBase}'s.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/Embedding">tf.keras.layers.Embedding</a>.
 */
public class EmbeddingWeight extends EyeBase {

  /** The weight function's parameters, past its function object. */
  protected enum Parameters {
    INPUT_DIM,
    OUTPUT_DIM;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public EmbeddingWeight(PointsToSetVariable source) {
    super(source);
  }

  public EmbeddingWeight(CGNode node) {
    super(node);
  }

  @Override
  protected int getNumRowsParameterPosition() {
    return Parameters.INPUT_DIM.getIndex();
  }

  @Override
  protected String getNumRowsParameterName() {
    return Parameters.INPUT_DIM.getName();
  }

  @Override
  protected int getNumColumnsParameterPosition() {
    return Parameters.OUTPUT_DIM.getIndex();
  }

  @Override
  protected String getNumColumnsParameterName() {
    return Parameters.OUTPUT_DIM.getName();
  }

  /**
   * An embedding weight's column count is the mandatory {@code output_dim}, with no default: when
   * the argument does not resolve, the column count is unknown, never the row count an identity
   * matrix would take.
   *
   * @param numRows The possible row counts, unused.
   * @return An unknown column count.
   */
  @Override
  protected Set<Optional<Integer>> getDefaultNumberOfColumns(Set<Optional<Integer>> numRows) {
    return Set.of(Optional.empty());
  }

  /**
   * The weight's dtype is the layer's: float32, the default this generator always reads. A {@code
   * dtype} keyword reaching the layer's base constructor would change it, and is not read.
   *
   * @return {@link #UNDEFINED_PARAMETER_POSITION}: no dtype parameter is read.
   */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * No dtype parameter is read; see {@link #getDTypeParameterPosition()}.
   *
   * @return {@code null}.
   */
  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
