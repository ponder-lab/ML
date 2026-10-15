package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder.LIST_APPEND_CONTENTS_FIELD;
import static com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder.LIST_OPERATION_CONTENTS_FIELD;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.types.FieldReference;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.junit.Test;

/**
 * Where a list keeps an element added by {@code append}, by {@code +=} and by {@code extend}.
 *
 * <p>A client that walks a list's object catalog finds an element only under a field the catalog
 * names, so these assert on the catalog and on the two synthetic contents fields the list rules
 * write, not on a tensor type. Each function in the fixture builds a one-element list holding the
 * tuple {@code (v, x)}.
 */
public class TestListExtension extends TestPythonMLCallGraphShape {

  /** The fixture, which builds the same list three ways. */
  private static final String FIXTURE = "tf2_test_list_extension.py";

  /** The rendering of a field's contents when the field holds the tuple. */
  private static final Set<String> TUPLE = Set.of("Ltuple");

  /**
   * The baseline: {@code append} stores the tuple under the append-contents field and enters that
   * field in the list's catalog.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testAppendCatalogsItsContents() throws Exception {
    Map<String, Set<String>> list = returnedList("appended");
    assertEquals(
        Map.of(
            "catalog",
            Set.of(LIST_APPEND_CONTENTS_FIELD),
            LIST_APPEND_CONTENTS_FIELD,
            TUPLE,
            LIST_OPERATION_CONTENTS_FIELD,
            Set.of()),
        list);
  }

  /**
   * A list extended with {@code +=} holds the tuple under the operation-contents field, but its
   * catalog does not name that field, so a walk over the catalog finds no element.
   *
   * <p>TODO: Expect the catalog to name {@link
   * PythonSSAPropagationCallGraphBuilder#LIST_OPERATION_CONTENTS_FIELD} once <a
   * href="https://github.com/wala/ML/issues/1069">wala/ML#1069</a> is fixed.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testAddedListDoesNotCatalogItsContents() throws Exception {
    Map<String, Set<String>> list = returnedList("added");
    assertEquals(
        Map.of(
            "catalog",
            Set.of(),
            LIST_APPEND_CONTENTS_FIELD,
            Set.of(),
            LIST_OPERATION_CONTENTS_FIELD,
            TUPLE),
        list);
  }

  /**
   * A list extended with {@code extend} holds nothing: {@code extend} is not modeled, so neither
   * contents field holds the tuple and the catalog is empty.
   *
   * <p>TODO: Expect the tuple, under a cataloged field, once <a
   * href="https://github.com/wala/ML/issues/1070">wala/ML#1070</a> is fixed.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testExtendedListHoldsNothing() throws Exception {
    Map<String, Set<String>> list = returnedList("extended");
    assertEquals(
        Map.of(
            "catalog",
            Set.of(),
            LIST_APPEND_CONTENTS_FIELD,
            Set.of(),
            LIST_OPERATION_CONTENTS_FIELD,
            Set.of()),
        list);
  }

  /**
   * The catalog and the two contents fields of the list a named function returns, unioned over the
   * function's nodes and the list's allocations. Catalog entries are rendered as their constant
   * values and field contents as their allocated types.
   *
   * @param function The function's name.
   * @return The catalog under {@code "catalog"} and each contents field under its name.
   * @throws Exception On analysis failure.
   */
  private Map<String, Set<String>> returnedList(String function) throws Exception {
    PythonTensorAnalysisEngine engine = (PythonTensorAnalysisEngine) makeEngine(FIXTURE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();

    Set<InstanceKey> lists = new java.util.HashSet<>();
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().contains("." + function + ".")) continue;
      for (SSAInstruction instruction : node.getIR().getInstructions())
        if (instruction instanceof SSAReturnInstruction ret && ret.getResult() > 0)
          pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, ret.getResult()))
              .forEach(lists::add);
    }
    assertFalse("The function returns no list: " + function, lists.isEmpty());

    Map<String, Set<String>> ret = new TreeMap<>();
    Set<String> catalog = new TreeSet<>();
    for (InstanceKey list : lists)
      for (InstanceKey entry :
          pa.getPointsToSet(
              ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                  .getPointerKeyForObjectCatalog(list)))
        catalog.add(
            entry instanceof ConstantKey<?> constant
                ? String.valueOf(constant.getValue())
                : entry.toString());
    ret.put("catalog", catalog);
    for (String name : new String[] {LIST_APPEND_CONTENTS_FIELD, LIST_OPERATION_CONTENTS_FIELD}) {
      Set<String> contents = new TreeSet<>();
      IField field =
          builder
              .getClassHierarchy()
              .resolveField(
                  FieldReference.findOrCreate(
                      PythonTypes.Root, Atom.findOrCreateAsciiAtom(name), PythonTypes.Root));
      if (field != null)
        for (InstanceKey list : lists)
          for (InstanceKey element :
              pa.getPointsToSet(builder.getPointerKeyForInstanceField(list, field)))
            contents.add(element.concreteType().getName().toString());
      ret.put(name, contents);
    }
    return ret;
  }
}
