package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by the <code>tf.keras.Model()</code> function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model">tf.keras.Model</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Model extends TensorGenerator {

  /** Parameter positions and names for {@code tf.keras.Model}. */
  protected enum Parameters {
    /** The input(s) of the model. */
    INPUTS,
    /** The output(s) of the model. */
    OUTPUTS,
    /** Name of the model. */
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Model(PointsToSetVariable source) {
    super(source);
  }

  public Model(CGNode node) {
    super(node);
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

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return getWeightShapes(builder, this.getSource());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return getWeightDTypes(builder, this.getSource());
  }

  /**
   * Infers the shapes of the model's weights by traversing the dataflow graph.
   *
   * @param builder The propagation call graph builder.
   * @param modelSource The source points-to set variable representing the model.
   * @return A set of weight shapes.
   */
  public Set<List<Dimension<?>>> getWeightShapes(
      PropagationCallGraphBuilder builder, PointsToSetVariable modelSource) {
    Set<List<Dimension<?>>> weightShapes = HashSetFactory.make();
    CallGraph cg = builder.getCallGraph();

    // We want to collect all Dense information from the entire call graph.
    Set<Integer> unitsValues = HashSetFactory.make();
    Set<List<Dimension<?>>> inputShapes = HashSetFactory.make();

    for (CGNode node : cg) {
      IClass declaringClass = node.getMethod().getDeclaringClass();
      String className = declaringClass.getReference().getName().toString();
      String realClassName = className;
      if (declaringClass instanceof PythonInstanceMethodTrampoline) {
        realClassName =
            ((PythonInstanceMethodTrampoline) declaringClass).getRealClass().getName().toString();
      }

      if (realClassName.contains("Dense")) {
        DenseCall denseGen = new DenseCall(node);

        // Hyper-exhaustive search: check indices 0-5 for both units and inputs in node and all
        // callers.
        int[] indices = {0, 1, 2, 3, 4, 5};

        // Search current node
        for (int idx : indices) {
          OrdinalSet<InstanceKey> nodeUnitsPts =
              denseGen.getArgumentPointsToSet(builder, idx, "units");
          if (nodeUnitsPts != null && !nodeUnitsPts.isEmpty()) {
            for (InstanceKey ik : nodeUnitsPts) {
              if (ik instanceof ConstantKey) {
                Object val = ((ConstantKey<?>) ik).getValue();
                if (val instanceof Number) unitsValues.add(((Number) val).intValue());
              }
            }
          }

          OrdinalSet<InstanceKey> nodeInputPts =
              denseGen.getArgumentPointsToSet(builder, idx, "inputs");
          if (nodeInputPts != null && !nodeInputPts.isEmpty()) {
            inputShapes.addAll(denseGen.getShapesOfValue(builder, nodeInputPts));
          }
        }

        // Search all callers
        Iterator<CGNode> callers = cg.getPredNodes(node);
        while (callers.hasNext()) {
          CGNode caller = callers.next();
          for (int idx : indices) {
            OrdinalSet<InstanceKey> callerUnitsPts =
                denseGen.getArgumentPointsToSet(builder, caller, idx, "units");
            if (callerUnitsPts != null && !callerUnitsPts.isEmpty()) {
              for (InstanceKey ik : callerUnitsPts) {
                if (ik instanceof ConstantKey) {
                  Object val = ((ConstantKey<?>) ik).getValue();
                  if (val instanceof Number) unitsValues.add(((Number) val).intValue());
                }
              }
            }

            OrdinalSet<InstanceKey> callerInputPts =
                denseGen.getArgumentPointsToSet(builder, caller, idx, "inputs");
            if (callerInputPts != null && !callerInputPts.isEmpty()) {
              inputShapes.addAll(denseGen.getShapesOfValue(builder, callerInputPts));
            }
          }
        }
      }
    }

    for (List<Dimension<?>> inputShape : inputShapes) {
      if (inputShape.isEmpty()) continue;
      Dimension<?> lastDim = inputShape.get(inputShape.size() - 1);
      for (Integer units : unitsValues) {
        // Kernel shape: (input_dim, units)
        List<Dimension<?>> kernelShape = new ArrayList<>();
        kernelShape.add(lastDim);
        kernelShape.add(new NumericDim(units));
        weightShapes.add(kernelShape);

        // Bias shape: (units,)
        List<Dimension<?>> biasShape = new ArrayList<>();
        biasShape.add(new NumericDim(units));
        weightShapes.add(biasShape);
      }
    }

    return weightShapes;
  }

  /**
   * Infers the dtypes of the model's weights.
   *
   * @param builder The propagation call graph builder.
   * @param modelSource The source points-to set variable representing the model.
   * @return A set of weight dtypes.
   */
  public Set<DType> getWeightDTypes(
      PropagationCallGraphBuilder builder, PointsToSetVariable modelSource) {
    // For now, we assume float32 for most Keras weights.
    return EnumSet.of(DType.FLOAT32);
  }
}
