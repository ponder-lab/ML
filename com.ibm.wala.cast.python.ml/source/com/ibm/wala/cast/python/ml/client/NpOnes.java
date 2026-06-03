package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for ndarrays created by NumPy's {@code ones()} function.
 *
 * <p>Structurally identical to {@link Ones} (the shape comes from the mandatory shape-tuple
 * argument and the dtype from the {@code dtype} argument), differing only in the default dtype:
 * NumPy defaults to {@code float64} whereas TensorFlow's {@code tf.ones} defaults to {@code
 * float32}.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.ones.html">numpy.ones()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpOnes extends Ones {

  private static final Logger LOGGER = getLogger(NpOnes.class.getName());

  public NpOnes(PointsToSetVariable source) {
    super(source);
  }

  public NpOnes(CGNode node) {
    super(node);
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
                + source
                + ". Using NumPy default dtype of: "
                + FLOAT64
                + ".");

    return EnumSet.of(FLOAT64);
  }
}
