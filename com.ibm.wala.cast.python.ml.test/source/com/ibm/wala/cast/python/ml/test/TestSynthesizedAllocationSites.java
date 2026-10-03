package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.NumpyTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The allocations the builder synthesizes at one instruction are distinct objects (wala/ML#1009). A
 * slice of an array allocates the slice and the methods attached to it at the slice's instruction;
 * identified by the program counter alone, they were one instance key whose type was whichever
 * registered first, so the slice's {@code tolist} could hold its {@code transpose} object, and
 * which one varied with iteration order from run to run.
 */
public class TestSynthesizedAllocationSites extends TestPythonMLCallGraphShape {

  @Test
  public void testSliceAndItsMethodsAreDistinctObjects() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_ndarray_slice_methods.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());

    // The types allocated at each instruction of the script's body.
    Map<Integer, Set<TypeReference>> typesBySite = HashMapFactory.make();
    for (InstanceKey key : builder.getPointerAnalysis().getInstanceKeys()) {
      if (!(key instanceof AllocationSiteInNode site)) continue;
      CGNode node = site.getNode();
      if (!node.getMethod()
          .getDeclaringClass()
          .getName()
          .toString()
          .endsWith("tf2_test_ndarray_slice_methods.py")) continue;
      typesBySite
          .computeIfAbsent(site.getSite().getProgramCounter(), pc -> HashSetFactory.make())
          .add(site.getSite().getDeclaredType());
    }

    TypeReference tolist = NumpyTypes.NDARRAY_ATTRIBUTES.get(NumpyTypes.TOLIST_METHOD_NAME);
    TypeReference transpose = NumpyTypes.NDARRAY_ATTRIBUTES.get("transpose");
    boolean found = false;
    for (Set<TypeReference> types : typesBySite.values())
      if (types.contains(NumpyTypes.NDARRAY_TYPE)
          && types.contains(tolist)
          && types.contains(transpose)) found = true;
    assertTrue(
        "Expecting an instruction that allocates an array and its distinct `tolist` and"
            + " `transpose` objects: "
            + typesBySite,
        found);
  }
}
