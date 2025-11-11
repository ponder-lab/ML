package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Gamma.Parameters.ALPHA;
import static com.ibm.wala.cast.python.ml.client.Gamma.Parameters.BETA;
import static com.ibm.wala.cast.python.ml.client.Gamma.Parameters.DTYPE;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * A representation of the `tf.random.gamma` API in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/random/gamma">tf.random.gamma</a>
 *     API.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Gamma extends Ones {

  private static final String FUNCTION_NAME = "tf.random.gamma()";

  enum Parameters {
    SHAPE,
    ALPHA,
    BETA,
    DTYPE,
    SEED,
    NAME
  }

  public Gamma(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE.ordinal();
  }

  protected int getAlphaParameterPosition() {
    return ALPHA.ordinal();
  }

  protected int getBetaParameterPosition() {
    return BETA.ordinal();
  }

  protected int getAlphaParameterValueNumber(PropagationCallGraphBuilder builder) {
    Set<Integer> numberOfPossiblePositionalArguments =
        this.getNumberOfPossiblePositionalArguments(builder);
    int alphaParameterPosition = this.getAlphaParameterPosition();

    if (!numberOfPossiblePositionalArguments.stream()
        .anyMatch(n -> n >= alphaParameterPosition + 1))
      throw new IllegalStateException(
          "Alpha parameter is mandatory and must be provided explicitly.");

    return this.getArgumentValueNumber(alphaParameterPosition);
  }

  protected int getBetaParameterValueNumber(PropagationCallGraphBuilder builder) {
    Set<Integer> numberOfPossiblePositionalArguments =
        this.getNumberOfPossiblePositionalArguments(builder);
    int betaParameterPosition = this.getBetaParameterPosition();

    if (!numberOfPossiblePositionalArguments.stream().anyMatch(n -> n >= betaParameterPosition + 1))
      return -1; // Beta parameter is optional.

    return this.getArgumentValueNumber(betaParameterPosition);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    Set<List<Dimension<?>>> shapes = super.getShapes(builder);

    if (shapes.isEmpty())
      throw new IllegalStateException("Cannot determine shape for mandatory shape parameter.");

    // Get the shape of the alpha parameter.
    Set<List<Dimension<?>>> alphaShapes =
        this.getShapes(builder, this.getAlphaParameterValueNumber(builder));

    if (alphaShapes.isEmpty())
      throw new IllegalStateException("Cannot determine shape for mandatory alpha parameter.");

    // If there is no beta parameter.
    if (this.getBetaParameterValueNumber(builder) < 0)
      // return shape `tf.concat([shape, tf.shape(alpha)], axis=0)`.
      shapes.forEach(
          shape -> {
            alphaShapes.forEach(
                alphaShape -> {
                  List<Dimension<?>> newShape = new ArrayList<>(shape);
                  newShape.addAll(alphaShape);
                  ret.add(newShape);
                });
          });
    else { // There is a beta parameter.
      // return shape `tf.concat([shape, tf.shape(alpha + beta)], axis=0)`.
      shapes.forEach(
          shape -> {
            // Get the shape of the beta parameter, which is optional.
            Set<List<Dimension<?>>> betaShapes =
                this.getShapes(builder, this.getBetaParameterValueNumber(builder));

            alphaShapes.forEach(
                aShape -> {
                  betaShapes.forEach(
                      bShape -> {
                        List<Dimension<?>> newShape = new ArrayList<>(shape);
                        // Here we assume that alphaShape and betaShape are compatible for
                        // broadcasting.
                        // In a complete implementation, we would need to handle broadcasting rules
                        // properly.
                        int maxLength = Math.max(aShape.size(), bShape.size());

                        for (int i = 0; i < maxLength; i++) {
                          Dimension<?> dim;

                          if (i < aShape.size() && i < bShape.size())
                            // Both shapes have this dimension, take the maximum.
                            dim = Dimension.max(aShape.get(i), bShape.get(i));
                          else if (i < aShape.size())
                            // Only alpha shape has this dimension.
                            dim = aShape.get(i);
                          else
                            // Only beta shape has this dimension.
                            dim = bShape.get(i);

                          newShape.add(dim);
                        }

                        ret.add(newShape);
                      });
                });
          });
    }

    return ret;
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }
}
