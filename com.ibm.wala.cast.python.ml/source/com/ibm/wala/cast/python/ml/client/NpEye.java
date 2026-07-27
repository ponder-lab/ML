package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code np.eye(N, M=None, k=0, dtype=float64)}: an identity matrix of shape ({@code
 * N}, {@code M} or {@code N}). Unlike {@code tf.eye}, there is no {@code batch_shape}, the {@code
 * k} diagonal offset does not affect the type, and the unspecified {@code dtype} default is NumPy's
 * {@code float64} rather than {@code float32}. The corpus idiom is one-hot encoding via fancy
 * indexing, {@code np.eye(n)[labels]}. See <a
 * href="https://github.com/wala/ML/issues/797">wala/ML#797</a>.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpEye extends EyeBase {

  private static final Logger LOGGER = getLogger(NpEye.class.getName());

  protected enum Parameters {
    N,
    M,
    K,
    DTYPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public NpEye(PointsToSetVariable source) {
    super(source);
  }

  public NpEye(CGNode node) {
    super(node);
  }

  @Override
  protected String getNumRowsParameterName() {
    return Parameters.N.getName();
  }

  @Override
  protected String getNumColumnsParameterName() {
    return Parameters.M.getName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }

  /**
   * {@inheritDoc}
   *
   * <p>NumPy defaults to {@code float64} when no {@code dtype} argument is supplied.
   *
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @return A singleton set containing the default NumPy dtype, {@code float64}.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    LOGGER.fine(
        () ->
            "No dtype specified for source: "
                + describe(this.getSource())
                + ". Using NumPy default dtype of: "
                + FLOAT64
                + ".");

    return EnumSet.of(FLOAT64);
  }

  /**
   * Returns the producing library of the modeled value: an {@code np.eye(...)} call, so the value
   * is an ndarray (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
