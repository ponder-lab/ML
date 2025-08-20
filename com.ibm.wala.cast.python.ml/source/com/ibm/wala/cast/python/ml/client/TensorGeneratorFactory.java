package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import java.util.logging.Logger;

public class TensorGeneratorFactory {

  private static final Logger LOGGER = Logger.getLogger(TensorGeneratorFactory.class.getName());

  /** https://www.tensorflow.org/api_docs/python/tf/ones. */
  private static final MethodReference ONES =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/ones")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/constant. */
  private static final MethodReference CONSTANT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/constant")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/range. */
  private static final MethodReference RANGE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/range")),
          AstMethodReference.fnSelector);

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
    else
      throw new IllegalArgumentException(
          "Unknown call: " + calledFunction + " for source: " + source + ".");
  }
}
