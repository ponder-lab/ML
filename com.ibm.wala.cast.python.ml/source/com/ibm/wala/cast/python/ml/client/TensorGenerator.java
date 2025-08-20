package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.D_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSORFLOW;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector.CALL_STRING;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.ContextItem;
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
import java.util.logging.Logger;

public abstract class TensorGenerator {

  protected static final Logger LOGGER = Logger.getLogger(TensorGenerator.class.getName());

  private static final MethodReference IMPORT =
      MethodReference.findOrCreate(
          TENSORFLOW,
          findOrCreateAsciiAtom("import"),
          Descriptor.findOrCreate(null, TENSORFLOW.getName()));

  protected PointsToSetVariable source;

  protected CGNode node;

  public TensorGenerator(PointsToSetVariable source, CGNode node) {
    this.source = source;
    this.node = node;
  }

  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> shapes = getShapes(builder);
    EnumSet<DType> dTypes = getDTypes(builder);

    Set<TensorType> ret = HashSetFactory.make();

    // Create a tensor type for each possible shape and dtype combination.
    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), dimensionList));

    return ret;
  }

  /**
   * Returns the possible shapes of the tensor returned by this generator.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return a set of shapes, where each shape is represented as a list of dimensions
   */
  protected abstract Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder);

  protected abstract EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder);

  protected EnumSet<DType> getDTypes(
      PropagationCallGraphBuilder builder, Iterable<InstanceKey> dTypePointsToSet) {
    EnumSet<DType> ret = EnumSet.noneOf(DType.class);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

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

    return ret;
  }

  protected EnumSet<DType> getDTypes(PropagationCallGraphBuilder builder) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    // The dtype is the second explicit argument.
    // FIXME: Handle keyword arguments.
    PointerKey dTypePointerKey = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, 3);
    OrdinalSet<InstanceKey> dTypePointsToSet = pointerAnalysis.getPointsToSet(dTypePointerKey);

    // If the argument dtype is not specified.
    if (dTypePointsToSet.isEmpty()) return getDefaultDTypes(builder);
    else
      // The dtype points-to set is non-empty, meaning that the dtype was explicitly set.
      return getDTypes(builder, dTypePointsToSet);
  }
}
