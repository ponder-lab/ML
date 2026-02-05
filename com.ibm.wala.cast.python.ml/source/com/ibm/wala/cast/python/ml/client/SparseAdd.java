package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

public class SparseAdd extends ElementWiseOperation {

  protected enum Parameters {
    A,
    B,
    THRESHOLD,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public SparseAdd(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getXParameterPosition() {
    return Parameters.A.getIndex();
  }

  @Override
  protected String getXParameterName() {
    return Parameters.A.getName();
  }

  @Override
  protected int getYParameterPosition() {
    return Parameters.B.getIndex();
  }

  @Override
  protected String getYParameterName() {
    return Parameters.B.getName();
  }

  @Override
  protected Set<List<Dimension<?>>> getShapesOfValue(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {

    if (valuePointsToSet == null || valuePointsToSet.isEmpty()) {
      return super.getShapesOfValue(builder, valuePointsToSet);
    }

    boolean hasSparseTensor = false;
    for (InstanceKey ik : valuePointsToSet) {
      if (ik.getConcreteType().getReference().equals(TensorFlowTypes.SPARSE_TENSOR_TYPE)) {
        hasSparseTensor = true;
        break;
      }
    }

    if (!hasSparseTensor) {
      return super.getShapesOfValue(builder, valuePointsToSet);
    }

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey ik : valuePointsToSet) {
      if (ik.getConcreteType().getReference().equals(TensorFlowTypes.SPARSE_TENSOR_TYPE)) {
        FieldReference denseShapeFieldRef =
            FieldReference.findOrCreate(Root, findOrCreateAsciiAtom("dense_shape"), Root);

        IField field = builder.getClassHierarchy().resolveField(denseShapeFieldRef);
        if (field != null) {
          PointerKey pk = builder.getPointerKeyForInstanceField(ik, field);
          OrdinalSet<InstanceKey> fieldPointsTo = pointerAnalysis.getPointsToSet(pk);
          ret.addAll(this.getShapesFromShapeArgument(builder, fieldPointsTo));
        }
      }
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(builder, this.getXParameterPosition(), getXParameterName());

    if (pointsToSet == null || pointsToSet.isEmpty()) {
      return super.getDefaultDTypes(builder);
    }

    boolean hasSparseTensor = false;
    for (InstanceKey ik : pointsToSet) {
      if (ik.getConcreteType().getReference().equals(TensorFlowTypes.SPARSE_TENSOR_TYPE)) {
        hasSparseTensor = true;
        break;
      }
    }

    if (!hasSparseTensor) {
      return super.getDefaultDTypes(builder);
    }

    Set<DType> ret = EnumSet.noneOf(DType.class);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey ik : pointsToSet) {
      if (ik.getConcreteType().getReference().equals(TensorFlowTypes.SPARSE_TENSOR_TYPE)) {
        FieldReference valuesFieldRef =
            FieldReference.findOrCreate(Root, findOrCreateAsciiAtom("values"), Root);

        IField field = builder.getClassHierarchy().resolveField(valuesFieldRef);
        if (field != null) {
          PointerKey pk = builder.getPointerKeyForInstanceField(ik, field);
          OrdinalSet<InstanceKey> fieldPointsTo = pointerAnalysis.getPointsToSet(pk);
          ret.addAll(this.getDTypesOfValue(builder, fieldPointsTo));
        }
      }
    }
    return ret;
  }
}
