package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
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

  /**
   * Retrieves the shapes of the dataset elements at a specific index within the output signature.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @param index The index within the output signature tuple.
   * @return A set of possible shapes for the element at the given index.
   */
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
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
            if (fieldIndex != null && fieldIndex == index) {
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
          }
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
    }
    return this.getShapes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * <p>For {@code tf.data.Dataset.from_generator}, shapes are inferred from the {@code
   * output_signature} or the legacy {@code output_shapes} arguments.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Strategy 1: Check output_signature.
    // This is the modern way to specify output properties in from_generator.
    OrdinalSet<InstanceKey> outputSignaturePts =
        this.getArgumentPointsToSet(
            builder, Parameters.OUTPUT_SIGNATURE.getIndex(), Parameters.OUTPUT_SIGNATURE.getName());

    if (outputSignaturePts != null && !outputSignaturePts.isEmpty()) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (InstanceKey ik : outputSignaturePts) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        // Case 1.1: Structured signature (tuple or list of specs).
        if (asin != null
            && (asin.getConcreteType().getReference().equals(tuple)
                || asin.getConcreteType().getReference().equals(list))) {
          // We extract shapes from each component of the structure and union them.
          // This results in the dataset elements being identified as having any of
          // the types/shapes present in the signature.
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
          // Case 1.2: Single spec object as signature.
          ret.addAll(this.getShapesFromShapeArgument(builder, Collections.singleton(ik)));
        }
      }
      // If we found any shapes in the output_signature, return them.
      if (!ret.isEmpty()) {
        return ret;
      }
    }

    // Strategy 2: Check output_shapes (legacy).
    OrdinalSet<InstanceKey> outputShapesPts =
        this.getArgumentPointsToSet(
            builder, Parameters.OUTPUT_SHAPES.getIndex(), Parameters.OUTPUT_SHAPES.getName());

    if (outputShapesPts != null && !outputShapesPts.isEmpty()) {
      Set<List<Dimension<?>>> ret = this.getShapesFromShapeArgument(builder, outputShapesPts);
      // If we found any shapes in the output_shapes, return them.
      if (!ret.isEmpty()) {
        return ret;
      }
    }

    // Default: No shape information could be extracted from generator arguments.
    return Collections.emptySet();
  }

  /**
   * Retrieves the dtypes of the dataset elements at a specific index within the output signature.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @param index The index within the output signature tuple.
   * @return A set of possible dtypes for the element at the given index.
   */
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
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
            if (fieldIndex != null && fieldIndex == index) {
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
          }
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
    }
    return this.getDTypes(builder);
  }

  /**
   * Retrieves the tensor types (shape and dtype combinations) of the dataset elements at a specific
   * index within the output signature.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @param index The index within the output signature tuple.
   * @return A set of possible tensor types for the element at the given index.
   */
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> shapes = this.getShapesForIndex(builder, index);
    Set<DType> dTypes = this.getDTypesForIndex(builder, index);

    Set<TensorType> ret = HashSetFactory.make();

    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), dimensionList));

    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * <p>For {@code tf.data.Dataset.from_generator}, dtypes are inferred from the {@code
   * output_signature} or the legacy {@code output_types} arguments.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Strategy 1: Check output_signature.
    // This is the modern way to specify output properties in from_generator.
    OrdinalSet<InstanceKey> outputSignaturePts =
        this.getArgumentPointsToSet(
            builder, Parameters.OUTPUT_SIGNATURE.getIndex(), Parameters.OUTPUT_SIGNATURE.getName());

    if (outputSignaturePts != null && !outputSignaturePts.isEmpty()) {
      Set<DType> ret = EnumSet.noneOf(DType.class);
      for (InstanceKey ik : outputSignaturePts) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        // Case 1.1: Structured signature (tuple or list of specs).
        if (asin != null
            && (asin.getConcreteType().getReference().equals(tuple)
                || asin.getConcreteType().getReference().equals(list))) {
          // Extract dtypes from each component and union them.
          // This results in the dataset elements being identified as having any of
          // the dtypes present in the signature.
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
          // Case 1.2: Single spec object as signature.
          ret.addAll(this.getDTypesFromDTypeArgument(builder, Collections.singleton(ik)));
        }
      }
      // If we found any dtypes in the output_signature, return them.
      if (!ret.isEmpty()) {
        return ret;
      }
    }

    // Strategy 2: Check output_types (legacy).
    OrdinalSet<InstanceKey> outputTypesPts =
        this.getArgumentPointsToSet(
            builder, Parameters.OUTPUT_TYPES.getIndex(), Parameters.OUTPUT_TYPES.getName());

    if (outputTypesPts != null && !outputTypesPts.isEmpty()) {
      Set<DType> ret = this.getDTypesFromDTypeArgument(builder, outputTypesPts);
      // If we found any dtypes in the output_types, return them.
      if (!ret.isEmpty()) {
        return ret;
      }
    }

    // Default: No dtype information found.
    return Collections.emptySet();
  }
}
