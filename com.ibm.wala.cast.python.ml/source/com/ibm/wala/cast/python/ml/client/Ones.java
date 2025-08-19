package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.D_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSORFLOW;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector.CALL_STRING;
import static java.util.Arrays.asList;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.ContextItem;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.types.Descriptor;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * A generator for tensors created by the `ones()` function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/ones">TensorFlow ones() API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Ones extends TensorGenerator {

  private static final MethodReference IMPORT =
      MethodReference.findOrCreate(
          TENSORFLOW,
          findOrCreateAsciiAtom("import"),
          Descriptor.findOrCreate(null, TENSORFLOW.getName()));

  public Ones(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    // This is a call to `ones()`. The shape is in the first explicit argument.
    // TODO: Handle keyword arguments.
    PointerKey shapePK = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, 2);

    for (InstanceKey shapeIK : pointerAnalysis.getPointsToSet(shapePK)) {
      AllocationSiteInNode asin = getAllocationSiteInNode(shapeIK);
      TypeReference reference = asin.getConcreteType().getReference();

      if (reference.equals(list)) { // TODO: This can also be a tuple of tensors.
        // We have a list of integers that represent the shape.
        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            pointerAnalysis.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        // We expect the object catalog to contain a list of integers. Each element in the array
        // corresponds to the set of possible dimensions for that index.
        @SuppressWarnings("unchecked")
        Set<Dimension<Integer>>[] possibleDimensions = new Set[objectCatalogPointsToSet.size()];

        for (InstanceKey catalogIK : objectCatalogPointsToSet) {
          ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
          Object constantKeyValue = constantKey.getValue();

          Integer fieldIndex = (Integer) constantKeyValue;

          FieldReference subscript =
              FieldReference.findOrCreate(
                  PythonTypes.Root, findOrCreateAsciiAtom(fieldIndex.toString()), PythonTypes.Root);

          IField f = builder.getClassHierarchy().resolveField(subscript);
          LOGGER.fine("Found field: " + f);

          // We can now get the pointer key for the instance field.
          PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);
          LOGGER.fine("Found pointer key for instance field: " + pointerKeyForInstanceField + ".");

          // Get the points-to set for the instance field.
          OrdinalSet<InstanceKey> instanceFieldPointsToSet =
              pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);
          LOGGER.fine("Points-to set for instance field: " + instanceFieldPointsToSet + ".");

          // If the instance field points to a constant, we can use it as the shape.
          // TODO: Is it possible to also do it for (simple) expressions?
          Set<Dimension<Integer>> tensorDimensions = HashSetFactory.make();

          for (InstanceKey instanceFieldIK : instanceFieldPointsToSet) {
            if (instanceFieldIK instanceof ConstantKey) {
              // We have a constant key.
              ConstantKey<?> instanceFieldConstant = (ConstantKey<?>) instanceFieldIK;
              Object instanceFieldValue = instanceFieldConstant.getValue();

              // We have a shape value.
              Long shapeValue = (Long) instanceFieldValue;
              LOGGER.fine(
                  "Found shape value: " + shapeValue + " for " + source.getPointerKey() + ".");

              Dimension<Integer> dimension = new NumericDim(shapeValue.intValue());

              LOGGER.fine("Adding dimension: " + dimension + ".");
              tensorDimensions.add(dimension);
            } else
              throw new IllegalStateException(
                  "Expected a constant key for instance field: "
                      + pointerKeyForInstanceField
                      + ", but got: "
                      + instanceFieldIK
                      + ".");
          }

          LOGGER.info(
              "Found possible shape dimensions: "
                  + tensorDimensions
                  + " for field: "
                  + pointerKeyForInstanceField
                  + " for source: "
                  + source
                  + ".");

          // Add the shape dimensions.
          assert possibleDimensions[fieldIndex] == null
              : "Duplicate field index: "
                  + fieldIndex
                  + " in object catalog: "
                  + objectCatalogPointsToSet
                  + ".";

          possibleDimensions[fieldIndex] = tensorDimensions;
          LOGGER.fine(
              "Added shape dimensions: "
                  + tensorDimensions
                  + " for field index: "
                  + fieldIndex
                  + ".");
        }

        for (int i = 0; i < possibleDimensions.length; i++)
          for (Dimension<Integer> iDim : possibleDimensions[i]) {
            @SuppressWarnings("unchecked")
            Dimension<Integer>[] dimensions = new Dimension[possibleDimensions.length];

            dimensions[i] = iDim;

            for (int j = 0; j < possibleDimensions.length; j++)
              if (i != j)
                for (Dimension<Integer> jDim : possibleDimensions[j]) dimensions[j] = jDim;

            ret.add(asList(dimensions));
          }
      } else
        throw new IllegalStateException(
            "Expected a " + PythonTypes.list + " for the shape, but got: " + reference + ".");
    }

    return ret;
  }

  @Override
  protected EnumSet<DType> getDTypes(PropagationCallGraphBuilder builder) {
    EnumSet<DType> ret = EnumSet.noneOf(DType.class);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    // The dtype is the second explicit argument.
    // FIXME: Handle keyword arguments.
    PointerKey dTypePointerKey = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, 3);
    OrdinalSet<InstanceKey> dTypePointsToSet = pointerAnalysis.getPointsToSet(dTypePointerKey);

    if (dTypePointsToSet.isEmpty()) {
      // Use the default dtype of float32.
      ret.add(FLOAT32);
      LOGGER.info(
          "No dtype specified for source: "
              + source
              + ". Using default dtype of: "
              + FLOAT32
              + " .");
    } else { // there's an explicit argument.
      for (InstanceKey dTypeIK : dTypePointsToSet) {
        IClass concreteType = dTypeIK.getConcreteType();
        TypeReference typeReference = concreteType.getReference();

        if (typeReference.equals(TensorFlowTypes.D_TYPE)) {
          // we have a dtype.
          // let's see if it's float32.
          Set<CGNode> importNodes = builder.getCallGraph().getNodes(IMPORT);

          // find the import node from this file.
          Optional<CGNode> importNode =
              importNodes.stream()
                  .filter(
                      in -> {
                        ContextItem contextItem = in.getContext().get(CALL_STRING);
                        CallString cs = (CallString) contextItem;

                        // We expect the first method in the call string to be the import.
                        assert cs.getMethods().length == 1
                            : "Expected a single method in the call string, but got: "
                                + cs.getMethods().length
                                + " for node: "
                                + in;

                        IMethod method = cs.getMethods()[0];

                        CallString nodeCS = (CallString) node.getContext().get(CALL_STRING);

                        // We expect the first method in the call string to be the import.
                        assert nodeCS.getMethods().length == 1
                            : "Expected a single method in the call string, but got: "
                                + nodeCS.getMethods().length
                                + " for node: "
                                + in;

                        return method.equals(nodeCS.getMethods()[0]);
                      })
                  .findFirst();

          InstanceKey tensorFlowIK =
              pointerAnalysis
                  .getHeapModel()
                  .getInstanceKeyForAllocation(
                      importNode.get(), NewSiteReference.make(0, TENSORFLOW));

          FieldReference float32 =
              FieldReference.findOrCreate(
                  PythonTypes.Root, findOrCreateAsciiAtom(FLOAT32.name().toLowerCase()), D_TYPE);

          IField float32Field = builder.getClassHierarchy().resolveField(float32);

          PointerKey float32PK =
              pointerAnalysis
                  .getHeapModel()
                  .getPointerKeyForInstanceField(tensorFlowIK, float32Field);

          for (InstanceKey float32IK : pointerAnalysis.getPointsToSet(float32PK))
            if (float32IK.equals(dTypeIK)) {
              ret.add(FLOAT32);
              LOGGER.info(
                  "Found dtype: "
                      + FLOAT32
                      + " for source: "
                      + source
                      + " from dType: "
                      + dTypeIK
                      + ".");
            } else throw new IllegalStateException("Unknown dtype: " + dTypeIK + ".");
        } else
          throw new IllegalStateException(
              "Expected a "
                  + TensorFlowTypes.D_TYPE
                  + " for the dtype, but got: "
                  + typeReference
                  + ".");
      }
    }

    return ret;
  }
}
