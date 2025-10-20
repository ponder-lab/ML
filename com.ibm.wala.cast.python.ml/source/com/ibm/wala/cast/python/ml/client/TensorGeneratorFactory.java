package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FILL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ONES;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RANGE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNIFORM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS_LIKE;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.types.TypeReference;
import java.util.logging.Logger;

/**
 * A factory for creating TensorGenerator instances based on the called TensorFlow function.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TensorGeneratorFactory {

  private static final Logger LOGGER = Logger.getLogger(TensorGeneratorFactory.class.getName());

  public static TensorGenerator getGenerator(PointsToSetVariable source) {
    // Get the pointer key for the source.
    PointerKey pointerKey = source.getPointerKey();

    LocalPointerKey localPointerKey = (LocalPointerKey) pointerKey;
    CGNode node = localPointerKey.getNode();

    TypeReference calledFunction = node.getMethod().getDeclaringClass().getReference();
    LOGGER.info("Getting tensor generator for call to: " + calledFunction.getName() + ".");

    if (calledFunction.equals(ONES.getDeclaringClass())) return new Ones(source, node);
    else if (calledFunction.equals(CONSTANT.getDeclaringClass())) return new Constant(source, node);
    else if (calledFunction.equals(RANGE.getDeclaringClass())) return new Range(source, node);
    else if (calledFunction.equals(UNIFORM.getDeclaringClass())) return new Uniform(source, node);
    else if (calledFunction.equals(ZEROS.getDeclaringClass())) return new Zeros(source, node);
    else if (calledFunction.equals(ZEROS_LIKE.getDeclaringClass()))
      return new ZerosLike(source, node);
    else if (calledFunction.equals(FILL.getDeclaringClass())) return new Fill(source, node);
    else
      throw new IllegalArgumentException(
          "Unknown call: " + calledFunction + " for source: " + source + ".");
  }
}
