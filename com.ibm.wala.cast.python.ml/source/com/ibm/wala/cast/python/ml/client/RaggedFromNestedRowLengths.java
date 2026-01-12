package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.RaggedFromNestedRowLengths.Parameters.NESTED_ROW_LENGTHS;
import static com.ibm.wala.cast.python.ml.client.RaggedFromNestedRowLengths.Parameters.VALUES;
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

  private static final String NESTED_ROW_LENGTHS_PARAM = "nested_row_lengths";

  protected enum Parameters {
    VALUES,
    NESTED_ROW_LENGTHS,
    NAME,
    VALIDATE
  }

  public RaggedFromNestedRowLengths(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getValuesParameterPosition() {
    return VALUES.ordinal();
  }

  protected int getNestedRowLengthsParameterPosition() {
    return NESTED_ROW_LENGTHS.ordinal();
  }

  protected OrdinalSet<InstanceKey> getNestedRowLengthsPointsToSet(
      PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(
        builder, this.getNestedRowLengthsParameterPosition(), NESTED_ROW_LENGTHS_PARAM);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // 1. Determine `nrows` and number of ragged dimensions from `nested_row_lengths`.
    // The number of rows is len(nested_row_lengths[0]).
    OrdinalSet<InstanceKey> nestedRowLengthsPts = this.getNestedRowLengthsPointsToSet(builder);
    Set<List<Dimension<?>>> nestedRowLengthsShapes = emptySet();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    Set<Dimension<?>> possibleRowDims = HashSetFactory.make();
    Set<Integer> possibleK = HashSetFactory.make(); // Number of ragged dimensions.

    if (nestedRowLengthsPts != null && !nestedRowLengthsPts.isEmpty()) {
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
            possibleK.add(k);

            // Get the first element of nested_row_lengths to determine nrows.
            if (k > 0) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      com.ibm.wala.cast.python.types.PythonTypes.Root,
                      Atom.findOrCreateAsciiAtom("0"),
                      com.ibm.wala.cast.python.types.PythonTypes.Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                OrdinalSet<InstanceKey> firstElemPts = pointerAnalysis.getPointsToSet(pk);
                Set<List<Dimension<?>>> shapesOfFirstElem =
                    this.getShapesOfValue(builder, firstElemPts);
                for (List<Dimension<?>> shape : shapesOfFirstElem) {
                  if (!shape.isEmpty()) {
                    possibleRowDims.add(shape.get(0));
                  }
                }
              }
            } else {
              // Should not happen for valid input? Or maybe 0 ragged dims?
              possibleRowDims.add(null);
            }
          }
        }
      }
    } else {
      possibleRowDims.add(null);
      possibleK.add(null);
    }

    // 2. Determine shape of `values`.
    OrdinalSet<InstanceKey> valuesPts =
        this.getArgumentPointsToSet(builder, getValuesParameterPosition(), VALUES_PARAM);
    Set<List<Dimension<?>>> valuesShapes = emptySet();
    if (valuesPts != null && !valuesPts.isEmpty()) {
      valuesShapes = this.getShapesOfValue(builder, valuesPts);
    } else {
      // Assume values can be anything if not known? Or handle empty?
      // If values points to empty, maybe it's just unknown.
      valuesShapes = new java.util.HashSet<>();
      valuesShapes.add(emptyList());
    }

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    for (Integer k : possibleK) {
      if (k == null) continue; // Skip if unknown K? Or assume something?

      for (Dimension<?> rowDim : possibleRowDims) {
        // If rowDim is null, it means we don't know the number of rows.
        // If K is 0, then we just have values? No, from_nested_row_lengths implies at least list of
        // lists?
        // Actually, if nested_row_lengths is empty list, it's identity?
        // TF docs say: "Returns a RaggedTensor with this.ragged_rank = nested_row_lengths.length"

        for (List<Dimension<?>> valShape : valuesShapes) {
          List<Dimension<?>> shape = new ArrayList<>();
          // First dimension is nrows (length of first list in nested_row_lengths)
          shape.add(rowDim);

          // Then K ragged dimensions (represented as null)
          // Wait, the first list in nested_row_lengths describes the splits for the FIRST ragged
          // dimension.
          // The output tensor has shape [nrows, (ragged), ..., (ragged), values.shape[1:]]
          // The number of ragged dimensions added is K.
          // So we add K nulls?
          // Let's verify with the test case: (4, None, None, None).
          // nested_row_lengths has 2 lists. K=2.
          // Result has 4 dimensions.
          // [nrows, null, null, values_dim_1] ?
          // In test: values is (None, None). values.shape[1:] is (None).
          // So [4, null, null, null].
          // That matches.
          // So we add K nulls.

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
      // Default fallback
      return getDefaultShapes(builder);
    }

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
