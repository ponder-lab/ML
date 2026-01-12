package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.RaggedFromNestedRowLengths.Parameters.FLAT_VALUES;
import static com.ibm.wala.cast.python.ml.client.RaggedFromNestedRowLengths.Parameters.NESTED_ROW_LENGTHS;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector.CALL_STRING;
import static java.util.Collections.emptyList;
import static java.util.Collections.emptySet;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of the `tf.RaggedTensor.from_nested_row_lengths` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_nested_row_lengths">tf.RaggedTensor.from_nested_row_lengths</a>.
 */
public class RaggedFromNestedRowLengths extends RaggedTensorFromValues {

  private static final Logger LOGGER = Logger.getLogger(RaggedFromNestedRowLengths.class.getName());

  protected enum Parameters {
    FLAT_VALUES,
    NESTED_ROW_LENGTHS,
    NAME,
    VALIDATE
  }

  public RaggedFromNestedRowLengths(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getValuesParameterPosition() {
    return FLAT_VALUES.ordinal();
  }

  @Override
  protected int getValuesArgumentValueNumber(PropagationCallGraphBuilder builder) {
    return this.getArgumentValueNumber(
        builder, this.getValuesParameterPosition(), FLAT_VALUES.name().toLowerCase(), true);
  }

  protected int getNestedRowLengthsParameterPosition() {
    return NESTED_ROW_LENGTHS.ordinal();
  }

  protected OrdinalSet<InstanceKey> getNestedRowLengthsPointsToSet(
      PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(
        builder,
        this.getNestedRowLengthsParameterPosition(),
        NESTED_ROW_LENGTHS.name().toLowerCase());
  }

  @Override
  protected OrdinalSet<InstanceKey> getArgumentPointsToSet(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {

    OrdinalSet<InstanceKey> combinedPts = OrdinalSet.empty();
    boolean found = false;

    // Analyze callers to find arguments passed (keyword or positional)
    CallString cs = (CallString) this.getNode().getContext().get(CALL_STRING);
    if (cs != null) {
      CallSiteReference siteReference = cs.getCallSiteRefs()[0];
      for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(this.getNode());
          it.hasNext(); ) {
        CGNode caller = it.next();
        SSAAbstractInvokeInstruction[] calls = caller.getIR().getCalls(siteReference);
        for (SSAAbstractInvokeInstruction callInstr : calls) {
          if (callInstr instanceof PythonInvokeInstruction) {
            PythonInvokeInstruction pyCallInstr = (PythonInvokeInstruction) callInstr;
            int argValNum = -1;

            // 1. Try keyword
            if (paramName != null) {
              argValNum = pyCallInstr.getUse(paramName);
              if (argValNum != -1) {
                LOGGER.info(
                    "Found keyword argument " + paramName + " with value number " + argValNum);
              }
            }

            // 2. Try positional if keyword not found
            if (argValNum == -1 && paramPos >= 0) {
              int numPosParams = pyCallInstr.getNumberOfPositionalParameters();
              if (paramPos + 1 < numPosParams) {
                argValNum = pyCallInstr.getUse(paramPos + 1);
              }
            }

            if (argValNum != -1) {
              PointerKey argPk =
                  builder
                      .getPointerAnalysis()
                      .getHeapModel()
                      .getPointerKeyForLocal(caller, argValNum);
              OrdinalSet<InstanceKey> argPts = builder.getPointerAnalysis().getPointsToSet(argPk);
              if (argPts != null) {
                combinedPts = OrdinalSet.unify(combinedPts, argPts);
                found = true;
              }
            }
          }
        }
      }
    }

    if (found) {
      return combinedPts;
    }

    return OrdinalSet.empty();
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    LOGGER.info("Calculating shapes for RaggedFromNestedRowLengths.");
    // 1. Determine `nrows` and number of ragged dimensions from `nested_row_lengths`.
    // The number of rows is len(nested_row_lengths[0]).
    OrdinalSet<InstanceKey> nestedRowLengthsPts = this.getNestedRowLengthsPointsToSet(builder);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    Set<Dimension<?>> possibleRowDims = HashSetFactory.make();
    Set<Integer> possibleK = HashSetFactory.make(); // Number of ragged dimensions.

    if (nestedRowLengthsPts != null && !nestedRowLengthsPts.isEmpty()) {
      LOGGER.info(
          "Found " + nestedRowLengthsPts.size() + " points-to set(s) for nested_row_lengths.");
      for (InstanceKey ik : nestedRowLengthsPts) {
        if (ik instanceof AllocationSiteInNode) {
          AllocationSiteInNode asin = getAllocationSiteInNode(ik);
          TypeReference reference = asin.getConcreteType().getReference();

          if (reference.equals(list) || reference.equals(tuple)) {
            OrdinalSet<InstanceKey> objectCatalogPointsToSet =
                pointerAnalysis.getPointsToSet(
                    ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                        .getPointerKeyForObjectCatalog(asin));
            int k = objectCatalogPointsToSet.size();
            LOGGER.info("Found nested_row_lengths list/tuple with length (K): " + k);
            possibleK.add(k);

            // Get the first element of nested_row_lengths to determine nrows.
            if (k > 0) {
              boolean foundRowDim = false;
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      com.ibm.wala.cast.python.types.PythonTypes.Root,
                      Atom.findOrCreateAsciiAtom("0"),
                      com.ibm.wala.cast.python.types.PythonTypes.Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                OrdinalSet<InstanceKey> firstElemPts = pointerAnalysis.getPointsToSet(pk);
                if (firstElemPts != null && !firstElemPts.isEmpty()) {
                  Set<List<Dimension<?>>> shapesOfFirstElem =
                      this.getShapesOfValue(builder, firstElemPts);
                  for (List<Dimension<?>> shape : shapesOfFirstElem) {
                    if (!shape.isEmpty()) {
                      LOGGER.info("Found row dimension from first element: " + shape.get(0));
                      possibleRowDims.add(shape.get(0));
                      foundRowDim = true;
                    }
                  }
                }
              }
              if (!foundRowDim) {
                possibleRowDims.add(null);
              }
            } else {
              // Should not happen for valid input? Or maybe 0 ragged dims?
              possibleRowDims.add(null);
            }
          }
        }
      }
    }

    if (possibleK.isEmpty()) {
      possibleK.add(null);
    }
    if (possibleRowDims.isEmpty()) {
      possibleRowDims.add(null);
    }

    // 2. Determine shape of `values`.
    OrdinalSet<InstanceKey> valuesPts =
        this.getArgumentPointsToSet(
            builder, getValuesParameterPosition(), FLAT_VALUES.name().toLowerCase());
    Set<List<Dimension<?>>> valuesShapes = emptySet();
    if (valuesPts != null && !valuesPts.isEmpty()) {
      valuesShapes = this.getShapesOfValue(builder, valuesPts);
      LOGGER.info("Found value shapes: " + valuesShapes);
    } else {
      valuesShapes = new java.util.HashSet<>();
      valuesShapes.add(emptyList());
      LOGGER.info("No value shapes found, assuming empty list.");
    }

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    for (Integer k : possibleK) {
      if (k == null) continue;

      for (Dimension<?> rowDim : possibleRowDims) {
        for (List<Dimension<?>> valShape : valuesShapes) {
          List<Dimension<?>> shape = new ArrayList<>();
          // First dimension is nrows (length of first list in nested_row_lengths)
          shape.add(rowDim);

          // Then K ragged dimensions (represented as null)
          for (int i = 0; i < k; i++) {
            shape.add(null);
          }

          // Then add values.shape[1:]
          if (valShape.size() > 1) {
            shape.addAll(valShape.subList(1, valShape.size()));
          }
          ret.add(shape);
        }
      }
    }

    // Handle case where we didn't find any shapes (e.g. points to sets empty)
    if (ret.isEmpty()) {
      throw new IllegalStateException("Could not calculate shapes for RaggedFromNestedRowLengths");
    }

    LOGGER.info("Final calculated shapes: " + ret);
    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return emptySet();
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }
}
