package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

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
    PythonTensorAnalysisEngine engine = (PythonTensorAnalysisEngine) makeEngine(file);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    Set<Object> ret = new HashSet<>();
    boolean found = false;
    for (CGNode node : cg) {
      String type = node.getMethod().getDeclaringClass().getName().toString();
      if (!type.startsWith("Lscript ") || !type.endsWith("/" + function)) continue;
      IR ir = node.getIR();
      for (int i = 0; i < ir.getNumberOfParameters(); i++) {
        int valueNumber = ir.getParameter(i);
        String[] names = ir.getLocalNames(0, valueNumber);
        if (names == null || !Arrays.asList(names).contains(parameter)) continue;
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
    assertFalse(function + " has no parameter " + parameter, !found);
    return ret;
  }
}
