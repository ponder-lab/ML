package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
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
   * A {@code Sequential} called with {@code training=True} passes it to each layer by keyword (<a
   * href="https://github.com/wala/ML/issues/1120">wala/ML#1120</a>), so it binds to a layer's
   * {@code training} wherever the parameter is, and a layer's {@code mask} keeps its default.
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
   * A {@code Sequential}'s layer whose {@code call} takes no {@code training} is still called with
   * its input.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testSequentialNoParameter() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_sequential_training.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    boolean reached = false;
    for (CGNode node : cg) {
      String type = node.getMethod().getDeclaringClass().getName().toString();
      if (!type.equals("Lscript tf2_test_sequential_training.py/Plain/call")) continue;
      for (int valueNumber : node.getIR().getParameterValueNumbers()) {
        String[] names = node.getIR().getLocalNames(0, valueNumber);
        if (names == null || !Arrays.asList(names).contains("x")) continue;
        PointerKey input =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, valueNumber);
        reached |= !builder.getPointerAnalysis().getPointsToSet(input).isEmpty();
      }
    }
    assertTrue("The input doesn't reach `Plain.call`.", reached);
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
