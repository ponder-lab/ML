package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.RaggedFromNestedRowLengths.Parameters.FLAT_VALUES;
import static com.ibm.wala.cast.python.ml.client.RaggedFromNestedRowLengths.Parameters.NESTED_ROW_LENGTHS;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static java.util.Collections.emptyList;
import static java.util.Collections.emptySet;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
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
    VALIDATE;

    public String getName() {
      return name().toLowerCase();
    }
  }

  public RaggedFromNestedRowLengths(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getValuesParameterPosition() {
    return FLAT_VALUES.ordinal();
  }

  @Override
  protected String getValuesParameterName() {
    return FLAT_VALUES.getName();
  }

  protected int getNestedRowLengthsParameterPosition() {
    return NESTED_ROW_LENGTHS.ordinal();
  }

  protected String getNestedRowLengthsParameterName() {
    return NESTED_ROW_LENGTHS.getName();
  }

  protected OrdinalSet<InstanceKey> getNestedStructurePointsToSet(
      PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(
        builder, this.getNestedRowLengthsParameterPosition(), getNestedRowLengthsParameterName());
  }

  protected Dimension<?> computeRowDim(Dimension<?> dim) {
    return dim;
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    LOGGER.info("Calculating shapes for RaggedFromNestedRowLengths.");
    // 1. Determine `nrows` and number of ragged dimensions from `nested_row_lengths`.
    // The number of rows is len(nested_row_lengths[0]).
    OrdinalSet<InstanceKey> nestedRowLengthsPts = this.getNestedStructurePointsToSet(builder);
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
                      Dimension<?> dim = shape.get(0);
                      LOGGER.info("Found row dimension from first element: " + dim);
                      possibleRowDims.add(computeRowDim(dim));
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
            builder, getValuesParameterPosition(), getValuesParameterName());
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
