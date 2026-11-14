package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.python.util.ContainerMutations;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.IR;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.IntStream;
import org.junit.Test;

/**
 * The {@code training} value Keras's {@code fit}, {@code evaluate} and {@code predict} pass a
 * model's {@code call} (<a href="https://github.com/wala/ML/issues/1248">wala/ML#1248</a>): {@code
 * True} from {@code train_step} and {@code False} from {@code test_step} and {@code predict_step},
 * by keyword, so it binds to the parameter named {@code training} wherever it is.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestKerasTrainingArgument extends TestPythonMLCallGraphShape {

  /** The fixture of the call-context cases (wala/ML#1249). */
  private static final String CONTEXT = "tf2_test_training_context.py";

  /** How {@link #parameterValues} reports a {@code bool} of unknown value. */
  private static final String BOOL = "<bool>";

  /**
   * {@code fit} passes {@code training=True}.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testFit() throws Exception {
    assertEquals(
        Set.of(Boolean.TRUE),
        parameterConstants("tf2_test_training_fit.py", "Net/call", "training"));
  }

  /**
   * {@code evaluate} passes {@code training=False}.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testEvaluate() throws Exception {
    assertEquals(
        Set.of(Boolean.FALSE),
        parameterConstants("tf2_test_training_evaluate.py", "Net/call", "training"));
  }

  /**
   * {@code predict} passes {@code training=False}.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testPredict() throws Exception {
    assertEquals(
        Set.of(Boolean.FALSE),
        parameterConstants("tf2_test_training_predict.py", "Net/call", "training"));
  }

  /**
   * {@code training} binds by name, so a {@code call} that declares {@code mask} before it gets
   * {@code training=True} there and keeps {@code mask}'s default.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testAfterMask() throws Exception {
    assertEquals(
        Set.of(Boolean.TRUE),
        parameterConstants("tf2_test_training_after_mask.py", "Net/call", "training"));
    assertEquals(
        Set.of("None"), parameterConstants("tf2_test_training_after_mask.py", "Net/call", "mask"));
  }

  /**
   * A {@code call} that takes {@code **kwargs} and no {@code training} finds {@code training=True}
   * among them, as Keras passes it there.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testKeywordArguments() throws Exception {
    assertEquals(
        Set.of(Boolean.TRUE),
        localConstants("tf2_test_training_kwargs.py", "Net/call", "training", false));
  }

  /**
   * A {@code call} that declares no {@code training} is still called, its input passed, as the
   * keyword binds to nothing there.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testNoParameter() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_training_no_parameter.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    boolean reached = false;
    for (CGNode node : cg) {
      String type = node.getMethod().getDeclaringClass().getName().toString();
      if (!type.equals("Lscript tf2_test_training_no_parameter.py/model_input")) continue;
      PointerKey input = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
      reached |= !builder.getPointerAnalysis().getPointsToSet(input).isEmpty();
    }
    assertFalse("The input of `call` doesn't reach `model_input`.", !reached);
  }

  /**
   * A {@code Sequential} called with {@code training=True} gives each layer {@code True} (<a
   * href="https://github.com/wala/ML/issues/1120">wala/ML#1120</a>), through Keras's call context
   * (<a href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>), wherever the layer's
   * {@code training} parameter is, and in place of its default; a layer's {@code mask} keeps its
   * default.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testSequential() throws Exception {
    String file = "tf2_test_sequential_training.py";
    assertEquals(Set.of(Boolean.TRUE), parameterConstants(file, "Block/call", "training"));
    assertEquals(Set.of(Boolean.TRUE), parameterConstants(file, "Masked/call", "training"));
    assertEquals(Set.of("None"), parameterConstants(file, "Masked/call", "mask"));
  }

  /**
   * A {@code Sequential} called with {@code training=True} twice gives its layer {@code True}
   * alone, and one called with {@code False} and then {@code True} gives its layer both (<a
   * href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testSequentialSwitched() throws Exception {
    String file = "tf2_test_sequential_training_switched.py";
    assertEquals(Set.of(Boolean.TRUE), parameterConstants(file, "TrainedGate/call", "training"));
    assertEquals(
        Set.of(Boolean.FALSE, Boolean.TRUE),
        parameterConstants(file, "SwitchedGate/call", "training"));
  }

  /**
   * A {@code Sequential} called with a {@code training} the analysis cannot read gives its layer
   * any {@code bool}, and, as the value may be {@code None}, the layer's default, rather than the
   * default alone (<a href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testSequentialUnknown() throws Exception {
    Set<Object> values =
        parameterValues("tf2_test_training_context_unknown.py", "Gate/call", "training");
    assertTrue(values + " has no bool of unknown value", values.contains(BOOL));
  }

  /**
   * A layer called without {@code training} inside a model's {@code call} running with {@code
   * training=True} takes {@code True} from Keras's call context, not its default (<a
   * href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testContextNested() throws Exception {
    assertEquals(Set.of(Boolean.TRUE), parameterConstants(CONTEXT, "Nested/call", "training"));
  }

  /**
   * A layer called with {@code training=False} inside a call running with {@code True} keeps the
   * value it is passed (<a href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testContextExplicit() throws Exception {
    assertEquals(Set.of(Boolean.FALSE), parameterConstants(CONTEXT, "Explicit/call", "training"));
  }

  /**
   * A layer called through a helper function inside a model's {@code call} takes the model's {@code
   * training}, as the helper runs in the model's call context (<a
   * href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testContextHelper() throws Exception {
    assertEquals(Set.of(Boolean.TRUE), parameterConstants(CONTEXT, "Helped/call", "training"));
  }

  /**
   * A layer called inside a layer whose {@code call} declares no {@code training} takes the outer
   * model's, which the middle layer's call keeps as its context (<a
   * href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testContextThroughNoTraining() throws Exception {
    assertEquals(Set.of(Boolean.TRUE), parameterConstants(CONTEXT, "Passed/call", "training"));
  }

  /**
   * A layer's {@code call} called directly, {@code self.inner.call(x)}, does not go through {@code
   * Layer.__call__}, so it takes its default and not the context's {@code True} (<a
   * href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testContextDirectCall() throws Exception {
    assertEquals(Set.of("None"), parameterConstants(CONTEXT, "Direct/call", "training"));
  }

  /**
   * A layer called with {@code training=None} in a model called without {@code training} takes its
   * default, as one not passed {@code training} does (<a
   * href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testContextExplicitNone() throws Exception {
    assertEquals(Set.of(Boolean.FALSE), parameterConstants(CONTEXT, "Defaulted/call", "training"));
  }

  /**
   * {@code fit} and {@code predict} call the model with {@code training} {@code True} and {@code
   * False}, and a layer the model's {@code call} calls without {@code training} takes each (<a
   * href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testContextFit() throws Exception {
    assertEquals(
        Set.of(Boolean.TRUE, Boolean.FALSE),
        parameterConstants("tf2_test_training_context_fit.py", "Inner/call", "training"));
  }

  /**
   * A learning phase the program sets is a {@code training} value of a layer called outside any
   * layer call, beside its default (<a
   * href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testLearningPhase() throws Exception {
    assertEquals(
        Set.of(1L, "None"),
        parameterValues("tf2_test_training_learning_phase.py", "Phased/call", "training"));
  }

  /**
   * A {@code Sequential}'s layer whose {@code call} takes no {@code training} is still called with
   * its input, and is given no {@code training}.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testSequentialNoParameter() throws Exception {
    String file = "tf2_test_sequential_training.py";
    assertTrue("The input doesn't reach `Plain.call`.", parameterReached(file, "Plain/call", "x"));
    assertFalse("`Plain.call` has a `training`.", parameterReached(file, "Plain/call", "training"));
  }

  /**
   * A {@code Sequential} called without {@code training} leaves each layer its own {@code training}
   * default, as Keras does, and the layer's {@code training} arm is still analyzed.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testSequentialAbsent() throws Exception {
    String file = "tf2_test_sequential_training_absent.py";
    assertEquals(Set.of(Boolean.FALSE), parameterConstants(file, "Block/call", "training"));
    assertTrue("The input doesn't reach `scale`.", parameterReached(file, "scale", "x"));
  }

  /**
   * A {@code Sequential} trained with {@code fit} and run with {@code predict} passes its layer the
   * {@code training} value each step passes it: {@code True}, then {@code False}.
   *
   * <p>TODO: Expect this to pass once <a
   * href="https://github.com/wala/ML/issues/833">wala/ML#833</a> is fixed. A {@code Sequential}'s
   * summary declares only {@code __call__} and {@code add}, and an instance call resolves its
   * method on the exact receiver class, so {@code fit} and {@code predict}, which {@code Model}
   * declares, resolve nothing on it and the layer is never called.
   *
   * @throws Exception On analysis failure.
   */
  @Test(expected = AssertionError.class)
  public void testSequentialFit() throws Exception {
    assertEquals(
        Set.of(Boolean.TRUE, Boolean.FALSE),
        parameterConstants("tf2_test_sequential_training_fit.py", "Block/call", "training"));
  }

  /**
   * A {@code training} value the analysis learns only once it is solved, the learning phase, picks
   * the method a layer calls, and the method extends a list a slice was taken of. The analysis
   * solves again to reach it, so the mutations it records for the list are those of the call graph
   * it finally builds, not those it read before that solve (<a
   * href="https://github.com/wala/ML/issues/1249">wala/ML#1249</a>).
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testTrainingPicksMutation() throws Exception {
    String file = "tf2_test_training_picks_mutation.py";
    PythonTensorAnalysisEngine engine = (PythonTensorAnalysisEngine) makeEngine(file);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    boolean mutating = false;
    Set<InstanceKey> lists = new HashSet<>();
    for (CGNode node : cg) {
      String type = node.getMethod().getDeclaringClass().getName().toString();
      mutating |= type.equals("Lscript " + file + "/Picker/mutating");
      if (!type.equals("Lscript " + file + "/consume")) continue;
      PointerKey items = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
      for (InstanceKey list : builder.getPointerAnalysis().getPointsToSet(items)) lists.add(list);
    }
    assertTrue("`Picker.mutating` is not reached.", mutating);
    assertFalse("No list reaches `consume`.", lists.isEmpty());
    ContainerMutations mutations = ContainerMutations.of(builder);
    for (InstanceKey list : lists)
      assertTrue(
          list + " is not recorded as extended.",
          mutations.allUnappended()
              || mutations
                  .unappended()
                  .contains(
                      builder.getPointerAnalysis().getInstanceKeyMapping().getMappedIndex(list)));
  }

  /**
   * The values a parameter of a function points to, over the function's nodes: each constant's
   * value, {@code None} as the string {@code "None"}, and {@link #BOOL} for a {@code bool} of
   * unknown value.
   *
   * @param file The Python file.
   * @param function The function, by the end of its class's name, such as {@code Net/call}.
   * @param parameter The parameter's name.
   * @return The values; any other member is its concrete type's name.
   * @throws Exception On analysis failure.
   */
  private Set<Object> parameterValues(String file, String function, String parameter)
      throws Exception {
    PythonTensorAnalysisEngine engine = (PythonTensorAnalysisEngine) makeEngine(file);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    Set<Object> ret = new HashSet<>();
    for (CGNode node : cg) {
      String type = node.getMethod().getDeclaringClass().getName().toString();
      if (!type.equals("Lscript " + file + "/" + function)) continue;
      for (int valueNumber : node.getIR().getParameterValueNumbers()) {
        String[] names = node.getIR().getLocalNames(0, valueNumber);
        if (names == null || !Arrays.asList(names).contains(parameter)) continue;
        PointerKey key =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, valueNumber);
        for (InstanceKey member : builder.getPointerAnalysis().getPointsToSet(key))
          if (member instanceof ConstantKey<?> constant)
            ret.add(constant.getValue() == null ? "None" : constant.getValue());
          else if (member.concreteType().getReference().equals(PythonTypes.PYTHON_BOOL))
            ret.add(BOOL);
          else ret.add(member.concreteType().getName().toString());
      }
    }
    return ret;
  }

  /**
   * Whether a parameter of a function points to anything, in some node of the function.
   *
   * @param file The Python file.
   * @param function The function, by the end of its class's name, such as {@code Net/call}.
   * @param parameter The parameter's name.
   * @return Whether the parameter's points-to set is non-empty in some node of the function.
   * @throws Exception On analysis failure.
   */
  private boolean parameterReached(String file, String function, String parameter)
      throws Exception {
    PythonTensorAnalysisEngine engine = (PythonTensorAnalysisEngine) makeEngine(file);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    for (CGNode node : cg) {
      String type = node.getMethod().getDeclaringClass().getName().toString();
      if (!type.equals("Lscript " + file + "/" + function)) continue;
      for (int valueNumber : node.getIR().getParameterValueNumbers()) {
        String[] names = node.getIR().getLocalNames(0, valueNumber);
        if (names == null || !Arrays.asList(names).contains(parameter)) continue;
        PointerKey key =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, valueNumber);
        if (!builder.getPointerAnalysis().getPointsToSet(key).isEmpty()) return true;
      }
    }
    return false;
  }

  /**
   * The constant values a parameter of a function points to, over the function's nodes, each {@code
   * None} as the string {@code "None"}.
   *
   * @param file The Python file.
   * @param function The function, by the end of its class's name, such as {@code Net/call}.
   * @param parameter The parameter's name.
   * @return The constants; a member that is not a constant fails the test.
   * @throws Exception On analysis failure.
   */
  private Set<Object> parameterConstants(String file, String function, String parameter)
      throws Exception {
    return localConstants(file, function, parameter, true);
  }

  /**
   * The constant values a local variable of a function points to, over the function's nodes, each
   * {@code None} as the string {@code "None"}.
   *
   * @param file The Python file.
   * @param function The function, by the end of its class's name, such as {@code Net/call}.
   * @param name The variable's name.
   * @param parameter Whether the variable is a parameter, or else any local of that name.
   * @return The constants; a member that is not a constant fails the test.
   * @throws Exception On analysis failure.
   */
  private Set<Object> localConstants(String file, String function, String name, boolean parameter)
      throws Exception {
    PythonTensorAnalysisEngine engine = (PythonTensorAnalysisEngine) makeEngine(file);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    Set<Object> ret = new HashSet<>();
    boolean found = false;
    for (CGNode node : cg) {
      String type = node.getMethod().getDeclaringClass().getName().toString();
      if (!type.startsWith("Lscript ") || !type.endsWith("/" + function)) continue;
      IR ir = node.getIR();
      int[] valueNumbers =
          parameter
              ? ir.getParameterValueNumbers()
              : IntStream.rangeClosed(1, ir.getSymbolTable().getMaxValueNumber()).toArray();
      for (int valueNumber : valueNumbers) {
        String[] names = ir.getLocalNames(0, valueNumber);
        if (names == null || !Arrays.asList(names).contains(name)) continue;
        found = true;
        PointerKey key =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, valueNumber);
        for (InstanceKey member : builder.getPointerAnalysis().getPointsToSet(key)) {
          assertFalse(member + " is not a constant", !(member instanceof ConstantKey<?>));
          Object value = ((ConstantKey<?>) member).getValue();
          ret.add(value == null ? "None" : value);
        }
      }
    }
    assertFalse(function + " has no " + (parameter ? "parameter " : "local ") + name, !found);
    return ret;
  }
}
