package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConcreteTypeKey;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.SSAContextInterpreter;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.junit.Test;

public class TestPrimitives extends TestJythonCallGraphShape {

  @Test
  public void testSource1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("prim1.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    CAstCallGraphUtil.AVOID_DUMP.set(false);
    CAstCallGraphUtil.dumpCG(
        (SSAContextInterpreter) builder.getContextInterpreter(), builder.getPointerAnalysis(), CG);
    // verifyGraphAssertions(CG, assertionsCalls1);
  }

  /**
   * A binary operation over two literals folds to its own value: a float to a {@link Double} rather
   * than a truncated integer. A comparison, and a float that overflows to infinity, stay unfolded
   * (<a href="https://github.com/wala/ML/issues/1143">wala/ML#1143</a>). The arguments of the calls
   * to {@code sink} in {@code fold1.py} are read in program order.
   */
  @Test
  public void testFloatFolding()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("fold1.py");
    CGNode script = getNodes(CG, "script fold1.py").iterator().next();
    IR ir = script.getIR();
    SymbolTable symtab = ir.getSymbolTable();
    // Each call to `sink` has its own context, so its own node.
    Set<SSAAbstractInvokeInstruction> calls = new HashSet<>();
    for (CGNode sink : getNodes(CG, "script fold1.py/sink"))
      CG.getPossibleSites(script, sink)
          .forEachRemaining(site -> calls.addAll(List.of(ir.getCalls(site))));
    List<SSAAbstractInvokeInstruction> ordered = new ArrayList<>(calls);
    ordered.sort(Comparator.comparingInt(SSAAbstractInvokeInstruction::iIndex));
    List<Object> arguments = new ArrayList<>();
    for (SSAAbstractInvokeInstruction call : ordered) {
      int argument = call.getUse(1);
      arguments.add(symtab.isConstant(argument) ? symtab.getConstantValue(argument) : "unfolded");
    }
    assertEquals(List.of(0.75, 0.1 + 0.2, 0.4 + 0.5, 2, "unfolded", "unfolded"), arguments);
  }

  /**
   * A Python scalar whose value is not a literal points to one type-level key per type: the result
   * of a scalar builtin, an element of a {@code range} and an index {@code enumerate} supplies. A
   * literal keeps its constant key, and a {@code sum}, whose type follows its elements, has none
   * (<a href="https://github.com/wala/ML/issues/1148">wala/ML#1148</a>). Each {@code sink_*}
   * function in {@code scalar_keys1.py} is read for what its parameter points to.
   */
  @Test
  public void testScalarKeys()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("scalar_keys1.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    Map<String, Set<String>> pointees = new TreeMap<>();
    for (String sink :
        List.of("len", "int", "float", "bool", "range", "enumerate", "literal", "sum")) {
      Set<String> kinds = new TreeSet<>();
      Collection<CGNode> nodes = getNodes(CG, "script scalar_keys1.py/sink_" + sink);
      // An unreached sink would read as an empty set, so its presence is asserted.
      assertFalse("sink_" + sink + " is unreached", nodes.isEmpty());
      for (CGNode node : nodes)
        // The parameter is value number 2; value number 1 is the function object.
        for (InstanceKey ik : pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, 2)))
          kinds.add(
              ik instanceof ConstantKey
                  ? "constant"
                  : ik instanceof ConcreteTypeKey
                      ? ik.concreteType().getName().toString()
                      : ik.getClass().getSimpleName());
      pointees.put(sink, kinds);
    }
    Map<String, Set<String>> expected = new TreeMap<>();
    expected.put("len", Set.of("Lint"));
    expected.put("int", Set.of("Lint"));
    expected.put("float", Set.of("Lfloat"));
    expected.put("bool", Set.of("Lbool"));
    expected.put("range", Set.of("Lint"));
    expected.put("enumerate", Set.of("Lint"));
    expected.put("literal", Set.of("constant"));
    expected.put("sum", Set.of());
    assertEquals(expected, pointees);
  }

  /**
   * The result of an operator over Python scalars points to the type-level key of its type: a loop
   * counter to its literal and an {@code int}, a comparison to a {@code bool}, and an arithmetic
   * operation with a {@code float} operand to a {@code float}. A division gives an {@code int} or a
   * {@code float}, since {@code //} and {@code /} share one operator in the IR. An operator over a
   * value of no scalar kind gives nothing (<a
   * href="https://github.com/wala/ML/issues/1149">wala/ML#1149</a>). Each {@code sink_*} function
   * in {@code scalar_ops1.py} is read for what its parameter points to.
   */
  @Test
  public void testScalarOperations()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("scalar_ops1.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    Map<String, Set<String>> pointees = new TreeMap<>();
    for (String sink :
        List.of(
            "counter",
            "compare",
            "scaled",
            "quotient",
            "floor",
            "negated",
            "not",
            "both",
            "unknown",
            "inverted",
            "shifted",
            "masked",
            "sign_of_flag",
            "negated_length")) {
      Set<String> kinds = new TreeSet<>();
      Collection<CGNode> nodes = getNodes(CG, "script scalar_ops1.py/sink_" + sink);
      // An unreached sink would read as an empty set, so its presence is asserted.
      assertFalse("sink_" + sink + " is unreached", nodes.isEmpty());
      for (CGNode node : nodes)
        // The parameter is value number 2; value number 1 is the function object.
        for (InstanceKey ik : pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, 2)))
          kinds.add(
              ik instanceof ConstantKey
                  ? "constant"
                  : ik instanceof ConcreteTypeKey
                      ? ik.concreteType().getName().toString()
                      : ik.getClass().getSimpleName());
      pointees.put(sink, kinds);
    }
    Map<String, Set<String>> expected = new TreeMap<>();
    expected.put("counter", Set.of("constant", "Lint"));
    expected.put("compare", Set.of("Lbool"));
    expected.put("scaled", Set.of("Lfloat"));
    expected.put("quotient", Set.of("Lfloat", "Lint"));
    expected.put("floor", Set.of("Lfloat", "Lint"));
    expected.put("negated", Set.of("Lint"));
    expected.put("not", Set.of("Lbool"));
    expected.put("both", Set.of("Lbool"));
    expected.put("unknown", Set.of());
    expected.put("inverted", Set.of("Lint"));
    expected.put("shifted", Set.of("Lint"));
    expected.put("masked", Set.of("Lint"));
    expected.put("sign_of_flag", Set.of("Lint"));
    expected.put("negated_length", Set.of("Lint"));
    assertEquals(expected, pointees);
  }
}
