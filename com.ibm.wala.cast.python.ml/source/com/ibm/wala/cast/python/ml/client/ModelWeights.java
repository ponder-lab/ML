package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector.CALL_STRING;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.debug.UnimplementedError;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import java.util.logging.Logger;

public class ModelWeights extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(ModelWeights.class.getName());

  public ModelWeights(PointsToSetVariable source) {
    super(source);
  }

  public ModelWeights(CGNode node) {
    super(node);
  }

  /** Resolves the producer node of an instance key by tracing it back through its creation site. */
  private static CGNode findProducerNode(InstanceKey ik) {
    AllocationSiteInNode asin = Util.getAllocationSiteInNode(ik);
    return (asin != null) ? asin.getNode() : null;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    Queue<InstanceKey> queue = new LinkedList<>();
    Set<InstanceKey> visited = HashSetFactory.make();

    if (this.getSource() != null) {
      // Look at the points-to set of the source to find where this ModelWeights object was created.
      for (InstanceKey ik :
          builder.getPointerAnalysis().getPointsToSet(this.getSource().getPointerKey())) {
        AllocationSiteInNode asin = Util.getAllocationSiteInNode(ik);
        if (asin != null) {
          CGNode node = asin.getNode();
          OrdinalSet<InstanceKey> outputsPts =
              this.getArgumentPointsToSet(
                  builder, Model.Parameters.OUTPUTS.getIndex(), Model.Parameters.OUTPUTS.getName());
          LOGGER.fine("ModelWeights outputsPts: " + outputsPts);

          if ((outputsPts == null || outputsPts.isEmpty())
              && node.getContext().get(CALL_STRING) != null) {
            // Exhaustive tracing: If points-to set is empty (e.g., due to functional application
            // routing bug),
            // trace back to the producer layer in the caller's IR.
            CallString cs = (CallString) node.getContext().get(CALL_STRING);
            for (int i = 0; i < cs.getCallSiteRefs().length; i++) {
              CallSiteReference siteReference = cs.getCallSiteRefs()[i];
              IMethod callerMethod = cs.getMethods()[i];
              for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(node);
                  it.hasNext(); ) {
                CGNode caller = it.next();
                if (!caller.getMethod().equals(callerMethod)) continue;
                for (SSAAbstractInvokeInstruction callInstr :
                    caller.getIR().getCalls(siteReference)) {
                  if (callInstr instanceof PythonInvokeInstruction pyCall) {
                    int argVn = pyCall.getUse(Model.Parameters.OUTPUTS.getName());
                    LOGGER.fine("ModelWeights exhaustive argVn: " + argVn);
                    if (argVn > 0) {
                      SSAInstruction def = caller.getDU().getDef(argVn);
                      LOGGER.fine("ModelWeights exhaustive def: " + def);
                      if (def instanceof SSAAbstractInvokeInstruction pyDef) {
                        int receiverVn = pyDef.getUse(0);
                        LOGGER.fine("ModelWeights exhaustive receiverVn: " + receiverVn);
                        if (receiverVn > 0) {
                          PointerKey receiverPk =
                              builder
                                  .getPointerAnalysis()
                                  .getHeapModel()
                                  .getPointerKeyForLocal(caller, receiverVn);
                          OrdinalSet<InstanceKey> receiverPts =
                              builder.getPointerAnalysis().getPointsToSet(receiverPk);
                          LOGGER.fine("ModelWeights exhaustive receiverPts: " + receiverPts);
                          for (InstanceKey receiverIk : receiverPts) {
                            queue.add(receiverIk);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }

          if (outputsPts != null) {
            for (InstanceKey outIk : outputsPts) {
              queue.add(outIk);
            }
          }
        }
      }
    }

    if (queue.isEmpty()) {
      // Fallback for direct calls if any.
      OrdinalSet<InstanceKey> outputsPts = this.getArgumentPointsToSet(builder, 2, "outputs");
      LOGGER.fine("ModelWeights fallback outputsPts: " + outputsPts);
      if (outputsPts != null) {
        for (InstanceKey ik : outputsPts) {
          queue.add(ik);
        }
      }
    }

    if (queue.isEmpty()) return Collections.emptySet();

    while (!queue.isEmpty()) {
      InstanceKey ik = queue.poll();
      if (!visited.add(ik)) continue;

      LOGGER.fine("ModelWeights processing ik: " + ik);

      // Attempt to resolve functional applications back to their producer layer.
      CGNode producerNode = findProducerNode(ik);
      if (producerNode != null) {
        LOGGER.fine("ModelWeights found producer node: " + producerNode);
      }

      AllocationSiteInNode asin = Util.getAllocationSiteInNode(ik);
      if (asin == null) continue;

      CGNode node = asin.getNode();
      int def = Util.findDefinition(node, asin);
      if (def == -1) continue;

      PointerKey defKey =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, def);
      PointsToSetVariable pts = null;
      try {
        if (!builder.getPropagationSystem().isImplicit(defKey)) {
          pts = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
        }
      } catch (UnimplementedError e) {
        // ignore
      }

      TensorGenerator gen = null;
      if (pts != null) {
        gen = TensorGeneratorFactory.getGenerator(pts, builder);
      }
      if (gen == null) {
        gen = TensorGenerator.createManualGenerator(node, builder);
      }

      // If we still didn't find a specialized generator, but we have a producer node,
      // try to use that node to create the generator.
      if (gen == null && producerNode != null) {
        gen = TensorGenerator.createManualGenerator(producerNode, builder);
      }

      LOGGER.fine("ModelWeights resolved generator: " + gen);

      if (gen == null) {
        // If it's a Tensor result of a functional application (__call__), the producer layer is
        // 'self'.
        if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/__call__")) {
          int selfVn = node.getIR().getParameter(0);
          PointerKey selfKey =
              builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, selfVn);
          for (InstanceKey selfIk : builder.getPointerAnalysis().getPointsToSet(selfKey)) {
            queue.add(selfIk);
          }
          continue;
        }
      }

      if (gen instanceof DelegatingTensorGenerator) {
        gen = ((DelegatingTensorGenerator) gen).getUnderlying();
      }

      if (gen instanceof Dense) {
        // Find input shape
        OrdinalSet<InstanceKey> inputPts =
            gen.getArgumentPointsToSet(
                builder, Dense.Parameters.INPUTS.getIndex(), Dense.Parameters.INPUTS.getName());
        Set<List<Dimension<?>>> inputShapes = gen.getShapesOfValue(builder, inputPts);

        // Find units
        OrdinalSet<InstanceKey> unitsPts =
            gen.getArgumentPointsToSet(
                builder, Dense.Parameters.UNITS.getIndex(), Dense.Parameters.UNITS.getName());
        Set<Integer> unitsValues = HashSetFactory.make();
        if (unitsPts != null) {
          for (InstanceKey uik : unitsPts) {
            if (uik instanceof ConstantKey) {
              Object val = ((ConstantKey<?>) uik).getValue();
              if (val instanceof Number) unitsValues.add(((Number) val).intValue());
            }
          }
        }

        for (List<Dimension<?>> inShape : inputShapes) {
          if (inShape.isEmpty()) continue;
          Dimension<?> inDim = inShape.get(inShape.size() - 1);
          for (Integer units : unitsValues) {
            List<Dimension<?>> kernelShape = new ArrayList<>();
            kernelShape.add(inDim);
            kernelShape.add(new NumericDim(units));
            ret.add(kernelShape);

            List<Dimension<?>> biasShape = new ArrayList<>();
            biasShape.add(new NumericDim(units));
            ret.add(biasShape);
          }
        }

        if (inputPts != null) {
          for (InstanceKey inputIk : inputPts) {
            queue.add(inputIk);
          }
        }
      } else if (gen instanceof Input) {
        // Reached input
      } else if (gen != null) {
        // Try to get input argument if possible, generic traverse
        OrdinalSet<InstanceKey> inputPts = gen.getArgumentPointsToSet(builder, 0, "inputs");
        if (inputPts != null) {
          for (InstanceKey inputIk : inputPts) {
            queue.add(inputIk);
          }
        }
        inputPts = gen.getArgumentPointsToSet(builder, 0, "x");
        if (inputPts != null) {
          for (InstanceKey inputIk : inputPts) {
            queue.add(inputIk);
          }
        }
      }
    }

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.FLOAT32);
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
