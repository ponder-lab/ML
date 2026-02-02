package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static java.util.Collections.emptySet;
import static java.util.Collections.singleton;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.classLoader.IField;
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
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of the `tf.RaggedTensor.from_value_rowids` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_value_rowids">tf.RaggedTensor.from_value_rowids</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class RaggedFromValueRowIds extends RaggedTensorFromValues {

  private static final Logger LOGGER = Logger.getLogger(RaggedFromValueRowIds.class.getName());

  protected enum Parameters {
    VALUES,
    VALUE_ROWIDS,
    NROWS,
    NAME,
    VALIDATE;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public RaggedFromValueRowIds(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getValuesParameterPosition() {
    return Parameters.VALUES.getIndex();
  }

  @Override
  protected String getValuesParameterName() {
    return Parameters.VALUES.getName();
  }

  protected int getValueRowidsParameterPosition() {
    return Parameters.VALUE_ROWIDS.getIndex();
  }

  protected String getValueRowidsParameterName() {
    return Parameters.VALUE_ROWIDS.getName();
  }

  protected Set<Long> getPossibleValueRowidsArguments(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getValueRowidsParameterPosition(), getValueRowidsParameterName());

    if (pointsToSet == null || pointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set in source: " + this.getSource() + ".");

    Set<Long> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey instanceKey : pointsToSet) {
      if (instanceKey instanceof ConstantKey) {
        ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
        Object value = constantKey.getValue();
        if (value instanceof Long) {
          ret.add((Long) value);
        } else if (value instanceof Integer) {
          ret.add(((Integer) value).longValue());
        }
      } else if (instanceKey instanceof AllocationSiteInNode) {
        AllocationSiteInNode asin = getAllocationSiteInNode(instanceKey);
        TypeReference reference = asin.getConcreteType().getReference();

        if (reference.equals(list) || reference.equals(tuple)) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              pointerAnalysis.getPointsToSet(
                  ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                      .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
            Object constantKeyValue = constantKey.getValue();
            Integer fieldIndex = null;
            if (constantKeyValue instanceof Integer) {
              fieldIndex = (Integer) constantKeyValue;
            } else if (constantKeyValue instanceof String) {
              fieldIndex = Integer.parseInt((String) constantKeyValue);
            }

            if (fieldIndex != null) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

              IField f = builder.getClassHierarchy().resolveField(subscript);
              PointerKey pointerKeyForInstanceField =
                  builder.getPointerKeyForInstanceField(asin, f);
              OrdinalSet<InstanceKey> instanceFieldPointsToSet =
                  pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);

              for (InstanceKey valIK : instanceFieldPointsToSet) {
                if (valIK instanceof ConstantKey) {
                  Object val = ((ConstantKey<?>) valIK).getValue();
                  if (val instanceof Long) {
                    ret.add((Long) val);
                  } else if (val instanceof Integer) {
                    ret.add(((Integer) val).longValue());
                  }
                }
              }
            }
          }
        }
      }
    }

    LOGGER.info(() -> "Possible value rowids for " + this.getSource() + ": " + ret + ".");
    return ret;
  }

  protected int getNrowsParameterPosition() {
    return Parameters.NROWS.getIndex();
  }

  protected String getNrowsParameterName() {
    return Parameters.NROWS.getName();
  }

  protected Set<Long> getPossibleNrowsArguments(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getNrowsParameterPosition(), getNrowsParameterName());
    return getPossibleLongValues(pointsToSet);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // 1. Determine `nrows`.
    Set<Long> nrowsArgs = this.getPossibleNrowsArguments(builder);

    // If nrows is not specified, then it is inferred from `value_rowids`.
    if (nrowsArgs.isEmpty()) {
      // It's `value_rowids[-1] + 1` if `value_rowids` is non-empty, and `0` otherwise.
      Set<Long> valueRowids = this.getPossibleValueRowidsArguments(builder);
      if (valueRowids.isEmpty()) nrowsArgs = singleton(0L);
      else {
        Long max = null;
        for (Long l : valueRowids)
          if (max == null) max = l;
          else if (l > max) max = l;

        nrowsArgs = singleton(max + 1);
      }
      final Set<Long> finalNrowsArgs = nrowsArgs;
      LOGGER.info(() -> "Inferred nrows for " + this.getSource() + ": " + finalNrowsArgs + ".");
    } else {
      final Set<Long> finalNrowsArgs = nrowsArgs;
      LOGGER.info(
          () -> "Found nrows arguments for " + this.getSource() + ": " + finalNrowsArgs + ".");
    }

    // For now, if nrows is missing, we might assume unknown or handle it if we can deduce from
    // value_rowids.
    // However, if we can't find it, we'll assume null (unknown).
    Set<Dimension<?>> possibleRowDims = HashSetFactory.make();
    if (!nrowsArgs.isEmpty()) {
      for (Long nrows : nrowsArgs) {
        if (nrows != null) {
          possibleRowDims.add(new NumericDim(nrows.intValue()));
        } else {
          possibleRowDims.add(null);
        }
      }
    } else {
      // Unknown rows
      possibleRowDims.add(null);
    }

    // 2. Determine shape of `values`.
    OrdinalSet<InstanceKey> valuesPts = this.getValuesPointsToSet(builder);
    // If we can't determine values shape, we just return [nrows, null] at least.
    Set<List<Dimension<?>>> valuesShapes = emptySet();
    if (valuesPts != null && !valuesPts.isEmpty()) {
      valuesShapes = this.getShapesOfValue(builder, valuesPts);
    }

    final Set<List<Dimension<?>>> finalValuesShapes = valuesShapes;
    LOGGER.info(
        () -> "Possible values shapes for " + this.getSource() + ": " + finalValuesShapes + ".");

    return constructRaggedShape(possibleRowDims, valuesShapes);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // This shouldn't really be called as getShapes overrides it, but satisfying the abstract
    // method.
    return emptySet();
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION; // No explicit shape argument
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION; // No explicit dtype argument
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
