package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static java.util.Collections.emptyList;
import static java.util.Collections.emptySet;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.types.PythonTypes;
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
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Base for {@code tf.RaggedTensor.from_nested_row_*} generators: the nested-ragged shape
 * construction from {@code flat_values} (position 0) plus a nested partitioning argument (position
 * 1). The concrete forms ({@code from_nested_row_lengths}, {@code from_nested_row_splits}) are
 * siblings &mdash; neither is a kind of the other &mdash; so the shared shape logic lives here and
 * each supplies its own partitioning-argument name and row-dimension computation. Replaces the
 * chained {@code RaggedFromNestedRowSplits extends RaggedFromNestedRowLengths} (<a
 * href="https://github.com/wala/ML/issues/514">wala/ML#514</a>).
 */
public abstract class RaggedFromNested extends RaggedTensorFromValues {

  private static final Logger LOGGER = Logger.getLogger(RaggedFromNested.class.getName());

  protected enum Parameters {
    FLAT_VALUES,
    NESTED_STRUCTURE,
    NAME,
    VALIDATE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }
  }

  public RaggedFromNested(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getValuesParameterPosition() {
    return Parameters.FLAT_VALUES.ordinal();
  }

  @Override
  protected String getValuesParameterName() {
    return Parameters.FLAT_VALUES.getName();
  }

  protected int getNestedStructureParameterPosition() {
    return Parameters.NESTED_STRUCTURE.ordinal();
  }

  /**
   * The keyword name of the nested partitioning argument (e.g. {@code nested_row_lengths} or {@code
   * nested_row_splits}).
   *
   * @return The partitioning-argument keyword name.
   */
  protected abstract String getNestedStructureParameterName();

  protected OrdinalSet<InstanceKey> getNestedStructurePointsToSet(
      PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(
        builder, this.getNestedStructureParameterPosition(), getNestedStructureParameterName());
  }

  /**
   * Converts the first-element dimension of the nested partitioning argument into the ragged
   * tensor's row ({@code nrows}) dimension. {@code from_nested_row_lengths} uses it directly;
   * {@code from_nested_row_splits} subtracts one ({@code row_splits} has {@code nrows + 1}
   * entries).
   *
   * @param dim The first-element dimension of the nested partitioning argument.
   * @return The row dimension, or {@code null} if it cannot be computed.
   */
  protected abstract Dimension<?> computeRowDim(Dimension<?> dim);

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    LOGGER.fine("Calculating shapes for " + this.getClass().getSimpleName() + ".");
    // 1. Determine `nrows` and number of ragged dimensions from `nested_row_lengths`.
    // The number of rows is len(nested_row_lengths[0]).
    OrdinalSet<InstanceKey> nestedRowLengthsPts = this.getNestedStructurePointsToSet(builder);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    Set<Dimension<?>> possibleRowDims = HashSetFactory.make();
    Set<Integer> possibleK = HashSetFactory.make(); // Number of ragged dimensions.

    if (nestedRowLengthsPts != null && !nestedRowLengthsPts.isEmpty()) {
      LOGGER.fine(
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
            LOGGER.fine("Found nested_row_lengths list/tuple with length (K): " + k);
            possibleK.add(k);

            // Get the first element of nested_row_lengths to determine nrows.
            if (k > 0) {
              boolean foundRowDim = false;
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      PythonTypes.Root, Atom.findOrCreateAsciiAtom("0"), PythonTypes.Root);
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
                      LOGGER.fine("Found row dimension from first element: " + dim);
                      possibleRowDims.add(computeRowDim(dim));
                      foundRowDim = true;
                    }
                  }
                }
              }
              if (!foundRowDim) {
                possibleRowDims.add(DynamicDim.INSTANCE);
              }
            } else {
              // Should not happen for valid input? Or maybe 0 ragged dims?
              possibleRowDims.add(DynamicDim.INSTANCE);
            }
          }
        }
      }
    }

    if (possibleK.isEmpty()) {
      possibleK.add(null);
    }
    if (possibleRowDims.isEmpty()) {
      possibleRowDims.add(DynamicDim.INSTANCE);
    }

    // 2. Determine shape of `values`.
    OrdinalSet<InstanceKey> valuesPts =
        this.getArgumentPointsToSet(
            builder, getValuesParameterPosition(), getValuesParameterName());
    Set<List<Dimension<?>>> valuesShapes = emptySet();
    if (valuesPts != null && !valuesPts.isEmpty()) {
      valuesShapes = this.getShapesOfValue(builder, valuesPts);
      LOGGER.fine("Found value shapes: " + valuesShapes);
    } else {
      valuesShapes = new java.util.HashSet<>();
      valuesShapes.add(emptyList());
      LOGGER.fine("No value shapes found, assuming empty list.");
    }

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    for (Integer k : possibleK) {
      if (k == null) continue;

      for (Dimension<?> rowDim : possibleRowDims) {
        for (List<Dimension<?>> valShape : valuesShapes) {
          List<Dimension<?>> shape = new ArrayList<>();
          // First dimension is nrows (length of first list in nested_row_lengths)
          shape.add(rowDim);

          // Then K ragged dimensions.
          for (int i = 0; i < k; i++) {
            shape.add(RaggedDim.INSTANCE);
          }

          // Then add values.shape[1:]
          if (valShape.size() > 1) {
            shape.addAll(valShape.subList(1, valShape.size()));
          }
          ret.add(shape);
        }
      }
    }

    // The shape rides on mandatory arguments (e.g. the nested row lengths and values); when they're
    // unresolvable the points-to sets are empty and no shape can be built. Floor to ⊤ (unknown
    // shape) rather than aborting the whole analysis. wala/ML#612.
    if (ret.isEmpty()) {
      LOGGER.fine(
          () ->
              "Could not calculate shapes for "
                  + this.getClass().getSimpleName()
                  + "; flooring to unknown (⊤).");
      return null;
    }

    LOGGER.fine("Final calculated shapes: " + ret);
    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
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
