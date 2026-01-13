package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.RaggedFromNestedValueRowIds.Parameters.FLAT_VALUES;
import static com.ibm.wala.cast.python.ml.client.RaggedFromNestedValueRowIds.Parameters.NESTED_NROWS;
import static com.ibm.wala.cast.python.ml.client.RaggedFromNestedValueRowIds.Parameters.NESTED_VALUE_ROWIDS;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static java.util.Collections.emptyList;
import static java.util.Collections.emptySet;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
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
 * A representation of the `tf.RaggedTensor.from_nested_value_rowids` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_nested_value_rowids">tf.RaggedTensor.from_nested_value_rowids</a>.
 */
public class RaggedFromNestedValueRowIds extends RaggedTensorFromValues {

  private static final Logger LOGGER =
      Logger.getLogger(RaggedFromNestedValueRowIds.class.getName());

  protected enum Parameters {
    FLAT_VALUES,
    NESTED_VALUE_ROWIDS,
    NESTED_NROWS,
    NAME,
    VALIDATE
  }

  public RaggedFromNestedValueRowIds(PointsToSetVariable source) {
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

  protected int getNestedValueRowIdsParameterPosition() {
    return NESTED_VALUE_ROWIDS.ordinal();
  }

  protected int getNestedNrowsParameterPosition() {
    return NESTED_NROWS.ordinal();
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    LOGGER.info("Calculating shapes for RaggedFromNestedValueRowIds.");
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    // 1. Determine `nrows`.
    Set<Long> nrowsArgs = HashSetFactory.make();

    // Check `nested_nrows` first.
    OrdinalSet<InstanceKey> nestedNrowsPts =
        this.getArgumentPointsToSet(
            builder, this.getNestedNrowsParameterPosition(), NESTED_NROWS.name().toLowerCase());

    if (nestedNrowsPts != null && !nestedNrowsPts.isEmpty()) {
      for (InstanceKey ik : nestedNrowsPts) {
        if (ik instanceof AllocationSiteInNode) {
          AllocationSiteInNode asin = getAllocationSiteInNode(ik);
          TypeReference reference = asin.getConcreteType().getReference();
          if (reference.equals(list) || reference.equals(tuple)) {
            // Get index 0
            FieldReference subscript =
                FieldReference.findOrCreate(Root, Atom.findOrCreateAsciiAtom("0"), Root);
            IField f = builder.getClassHierarchy().resolveField(subscript);
            if (f != null) {
              PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
              OrdinalSet<InstanceKey> firstElemPts = pointerAnalysis.getPointsToSet(pk);
              for (InstanceKey valKey : firstElemPts) {
                if (valKey instanceof ConstantKey) {
                  Object val = ((ConstantKey<?>) valKey).getValue();
                  if (val instanceof Integer) nrowsArgs.add(((Integer) val).longValue());
                  else if (val instanceof Long) nrowsArgs.add((Long) val);
                }
              }
            }
          }
        }
      }
    }

    // 2. Determine K (ragged rank) and infer nrows if needed.
    OrdinalSet<InstanceKey> nestedValueRowIdsPts =
        this.getArgumentPointsToSet(
            builder,
            this.getNestedValueRowIdsParameterPosition(),
            NESTED_VALUE_ROWIDS.name().toLowerCase());

    Set<Dimension<?>> possibleRowDims = HashSetFactory.make();
    Set<Integer> possibleK = HashSetFactory.make();

    if (nestedValueRowIdsPts != null && !nestedValueRowIdsPts.isEmpty()) {
      for (InstanceKey ik : nestedValueRowIdsPts) {
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

            // Determine nrows if not explicit.
            if (nrowsArgs.isEmpty()) {
              if (k > 0) {
                // Get first element of nested_value_rowids
                FieldReference subscript =
                    FieldReference.findOrCreate(Root, Atom.findOrCreateAsciiAtom("0"), Root);
                IField f = builder.getClassHierarchy().resolveField(subscript);
                if (f != null) {
                  PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                  OrdinalSet<InstanceKey> firstElemPts = pointerAnalysis.getPointsToSet(pk);

                  Long max = null;
                  boolean foundAny = false;

                  for (InstanceKey innerIk : firstElemPts) {
                    if (innerIk instanceof AllocationSiteInNode) {
                      AllocationSiteInNode innerAsin = getAllocationSiteInNode(innerIk);
                      TypeReference innerRef = innerAsin.getConcreteType().getReference();
                      if (innerRef.equals(list) || innerRef.equals(tuple)) {
                        OrdinalSet<InstanceKey> innerObjectCatalog =
                            pointerAnalysis.getPointsToSet(
                                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                                    .getPointerKeyForObjectCatalog(innerAsin));

                        for (InstanceKey catKey : innerObjectCatalog) {
                          ConstantKey<?> ck = (ConstantKey<?>) catKey;
                          Object idxObj = ck.getValue();
                          String idxStr = idxObj.toString();

                          FieldReference innerSub =
                              FieldReference.findOrCreate(
                                  Root, findOrCreateAsciiAtom(idxStr), Root);
                          IField innerF = builder.getClassHierarchy().resolveField(innerSub);
                          if (innerF != null) {
                            PointerKey valPk =
                                builder.getPointerKeyForInstanceField(innerAsin, innerF);
                            OrdinalSet<InstanceKey> valPts = pointerAnalysis.getPointsToSet(valPk);
                            for (InstanceKey valKey : valPts) {
                              if (valKey instanceof ConstantKey) {
                                Object val = ((ConstantKey<?>) valKey).getValue();
                                long lVal = -1;
                                if (val instanceof Integer) lVal = ((Integer) val).longValue();
                                else if (val instanceof Long) lVal = (Long) val;

                                if (val instanceof Integer || val instanceof Long) {
                                  if (max == null || lVal > max) max = lVal;
                                  foundAny = true;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                  if (max != null) {
                    possibleRowDims.add(new NumericDim(max.intValue() + 1));
                  } else if (!foundAny) {
                    // Empty list of rowids?
                    possibleRowDims.add(new NumericDim(0));
                  } else {
                    possibleRowDims.add(null);
                  }
                } else {
                  possibleRowDims.add(null);
                }
              } else {
                possibleRowDims.add(null);
              }
            } else {
              for (Long n : nrowsArgs) {
                if (n != null) possibleRowDims.add(new NumericDim(n.intValue()));
                else possibleRowDims.add(null);
              }
            }
          }
        }
      }
    }

    if (possibleK.isEmpty()) possibleK.add(null);
    if (possibleRowDims.isEmpty()) possibleRowDims.add(null);

    // Values shape
    OrdinalSet<InstanceKey> valuesPts =
        this.getArgumentPointsToSet(
            builder, getValuesParameterPosition(), FLAT_VALUES.name().toLowerCase());
    Set<List<Dimension<?>>> valuesShapes = emptySet();
    if (valuesPts != null && !valuesPts.isEmpty()) {
      valuesShapes = this.getShapesOfValue(builder, valuesPts);
    } else {
      valuesShapes = new java.util.HashSet<>();
      valuesShapes.add(emptyList());
    }

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    for (Integer k : possibleK) {
      if (k == null) continue;

      for (Dimension<?> rowDim : possibleRowDims) {
        for (List<Dimension<?>> valShape : valuesShapes) {
          List<Dimension<?>> shape = new ArrayList<>();
          // First dimension
          shape.add(rowDim);

          // Then K ragged dimensions
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

    if (ret.isEmpty()) {
      LOGGER.info("Could not calculate shapes for RaggedFromNestedValueRowIds");
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
