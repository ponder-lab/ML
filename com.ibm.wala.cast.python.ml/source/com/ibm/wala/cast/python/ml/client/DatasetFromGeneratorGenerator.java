package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/** A generator for tensors created by {@code tf.data.Dataset.from_generator}. */
public class DatasetFromGeneratorGenerator extends DatasetGenerator {

  protected enum Parameters {
    GENERATOR,
    OUTPUT_TYPES,
    OUTPUT_SHAPES,
    ARGS,
    OUTPUT_SIGNATURE,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      // +1 to skip 'self' which is the Dataset class
      return ordinal() + 1;
    }
  }

  public DatasetFromGeneratorGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetFromGeneratorGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Check output_signature first
    OrdinalSet<InstanceKey> outputSignaturePts =
        this.getArgumentPointsToSet(
            builder, Parameters.OUTPUT_SIGNATURE.getIndex(), Parameters.OUTPUT_SIGNATURE.getName());

    if (outputSignaturePts != null && !outputSignaturePts.isEmpty()) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (InstanceKey ik : outputSignaturePts) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null
            && (asin.getConcreteType().getReference().equals(tuple)
                || asin.getConcreteType().getReference().equals(list))) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            FieldReference subscript =
                FieldReference.findOrCreate(
                    Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
            IField f = builder.getClassHierarchy().resolveField(subscript);
            if (f != null) {
              PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
              ret.addAll(
                  this.getShapesFromShapeArgument(
                      builder, builder.getPointerAnalysis().getPointsToSet(pk)));
            }
          }
        } else {
          ret.addAll(this.getShapesFromShapeArgument(builder, Collections.singleton(ik)));
        }
      }
      return ret;
    }

    // Check output_shapes
    OrdinalSet<InstanceKey> outputShapesPts =
        this.getArgumentPointsToSet(
            builder, Parameters.OUTPUT_SHAPES.getIndex(), Parameters.OUTPUT_SHAPES.getName());

    if (outputShapesPts != null && !outputShapesPts.isEmpty()) {
      return this.getShapesFromShapeArgument(builder, outputShapesPts);
    }

    return Collections.emptySet();
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Check output_signature first
    OrdinalSet<InstanceKey> outputSignaturePts =
        this.getArgumentPointsToSet(
            builder, Parameters.OUTPUT_SIGNATURE.getIndex(), Parameters.OUTPUT_SIGNATURE.getName());

    if (outputSignaturePts != null && !outputSignaturePts.isEmpty()) {
      Set<DType> ret = EnumSet.noneOf(DType.class);
      for (InstanceKey ik : outputSignaturePts) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null
            && (asin.getConcreteType().getReference().equals(tuple)
                || asin.getConcreteType().getReference().equals(list))) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            FieldReference subscript =
                FieldReference.findOrCreate(
                    Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
            IField f = builder.getClassHierarchy().resolveField(subscript);
            if (f != null) {
              PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
              ret.addAll(
                  this.getDTypesFromDTypeArgument(
                      builder, builder.getPointerAnalysis().getPointsToSet(pk)));
            }
          }
        } else {
          ret.addAll(this.getDTypesFromDTypeArgument(builder, Collections.singleton(ik)));
        }
      }
      return ret;
    }

    // Check output_types
    OrdinalSet<InstanceKey> outputTypesPts =
        this.getArgumentPointsToSet(
            builder, Parameters.OUTPUT_TYPES.getIndex(), Parameters.OUTPUT_TYPES.getName());

    if (outputTypesPts != null && !outputTypesPts.isEmpty()) {
      return this.getDTypesFromDTypeArgument(builder, outputTypesPts);
    }

    return Collections.emptySet();
  }
}
