package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code tf.layers.dense}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/layers/dense">tf.layers.dense</a>
 */
public class Dense extends TensorGenerator {

  /** Parameter positions and names for {@code tf.layers.dense}. */
  protected enum Parameters {
    /** The input tensor. */
    INPUTS,
    /** Integer or Long, dimensionality of the output space. */
    UNITS;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Dense(PointsToSetVariable source) {
    super(source);
  }

  public Dense(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    String methodName = this.getNode().getMethod().getName().toString();

    // Adjust parameter positions based on method signature
    int inputPos = -1;
    int unitsPos = -1;

    if (methodName.equals("__call__") || methodName.equals("call")) {
      inputPos = 1; // func, self, inputs (so inputs is pos 1)
    } else if (methodName.equals("do")
        && !this.getNode().getMethod().getDeclaringClass().getName().toString().contains("call")) {
      unitsPos = 1; // self, units (so units is pos 1)
    }

    Set<List<Dimension<?>>> inputShapes = HashSetFactory.make();
    OrdinalSet<InstanceKey> inputPts =
        inputPos >= 0
            ? this.getArgumentPointsToSet(builder, inputPos, Parameters.INPUTS.getName())
            : null;

    if (inputPts != null && !inputPts.isEmpty()) {
      inputShapes.addAll(this.getShapesOfValue(builder, inputPts));
    } else {
      // Fallback: Read 'inputs' field from self
      OrdinalSet<InstanceKey> selfPts =
          this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, null);
      if (selfPts != null) {
        for (InstanceKey ik : selfPts) {
          com.ibm.wala.classLoader.IField inputsField =
              ik.getConcreteType()
                  .getField(com.ibm.wala.core.util.strings.Atom.findOrCreateUnicodeAtom("inputs"));
          if (inputsField != null) {
            OrdinalSet<InstanceKey> fPts =
                builder
                    .getPointerAnalysis()
                    .getPointsToSet(
                        builder
                            .getPointerKeyFactory()
                            .getPointerKeyForInstanceField(ik, inputsField));
            if (fPts != null && !fPts.isEmpty()) {
              inputShapes.addAll(this.getShapesOfValue(builder, fPts));
            }
          }
        }
      }
    }

    LOGGER.fine("Dense inputShapes: " + inputShapes);
    if (inputShapes.isEmpty()) return Collections.emptySet();

    Set<Integer> unitsValues = HashSetFactory.make();
    OrdinalSet<InstanceKey> unitsPts =
        unitsPos >= 0
            ? this.getArgumentPointsToSet(builder, unitsPos, Parameters.UNITS.getName())
            : null;

    if (unitsPts != null && !unitsPts.isEmpty()) {
      for (InstanceKey ik : unitsPts) {
        if (ik instanceof ConstantKey) {
          Object val = ((ConstantKey<?>) ik).getValue();
          if (val instanceof Number) unitsValues.add(((Number) val).intValue());
        }
      }
    } else {
      // Fallback: Read 'units' field from self
      OrdinalSet<InstanceKey> selfPts = null;
      if (methodName.equals("__call__") || methodName.equals("call")) {
        selfPts = this.getArgumentPointsToSet(builder, 0, null);
      } else {
        selfPts = this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, null);
      }

      if (selfPts != null) {
        for (InstanceKey ik : selfPts) {
          com.ibm.wala.classLoader.IField unitsField =
              ik.getConcreteType()
                  .getField(com.ibm.wala.core.util.strings.Atom.findOrCreateUnicodeAtom("units"));
          if (unitsField != null) {
            OrdinalSet<InstanceKey> fPts =
                builder
                    .getPointerAnalysis()
                    .getPointsToSet(
                        builder
                            .getPointerKeyFactory()
                            .getPointerKeyForInstanceField(ik, unitsField));
            if (fPts != null) {
              for (InstanceKey p : fPts) {
                if (p instanceof ConstantKey) {
                  Object val = ((ConstantKey<?>) p).getValue();
                  if (val instanceof Number) unitsValues.add(((Number) val).intValue());
                }
              }
            }
          }
        }
      }
    }

    LOGGER.fine("Dense unitsValues: " + unitsValues);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      if (inputShape.isEmpty()) continue;
      for (Integer units : unitsValues) {
        List<Dimension<?>> newShape = new ArrayList<>(inputShape);
        newShape.set(newShape.size() - 1, new NumericDim(units));
        ret.add(newShape);
      }
      if (unitsValues.isEmpty()) {
        // If units are not provided (e.g., Keras __call__ or Model.do), we treat it as a
        // pass-through.
        if (methodName.equals("__call__") || methodName.equals("call") || methodName.equals("do")) {
          ret.add(inputShape);
        } else {
          List<Dimension<?>> newShape = new ArrayList<>(inputShape);
          newShape.set(newShape.size() - 1, new NumericDim(-1)); // Unknown units
          ret.add(newShape);
        }
      }
    }

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Derive dtype from the input tensor.
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());
    if (inputPts.isEmpty()) return EnumSet.noneOf(DType.class);
    return this.getDTypesOfValue(builder, inputPts);
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
