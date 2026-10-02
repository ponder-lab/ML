package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ml.types.NumpyTypes;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.xml.parsers.DocumentBuilderFactory;
import org.junit.Test;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

/**
 * The methods the call-graph builder attaches to an array it allocates itself (a slice's or an
 * arithmetic result's) are the ones every summary allocator of an array attaches, with the same
 * summary classes (wala/ML#1009), so the two kinds of array dispatch alike. Per-instance
 * attachment, and this check with it, goes away with wala/ML#551.
 */
public class TestNdarrayAttributes {

  /** The summary files that allocate arrays. */
  private static final List<String> SUMMARIES = List.of("numpy.xml", "scipy.xml", "tensorflow.xml");

  /** The array type's name as the summaries write it. */
  private static final String NDARRAY = "Lnumpy/ndarray";

  /**
   * Every {@code <new>} of an array in the summaries attaches exactly {@link
   * NumpyTypes#NDARRAY_ATTRIBUTES}: each named attribute receives a fresh instance of the same
   * summary class.
   *
   * @throws Exception On failure to read a summary.
   */
  @Test
  public void testEveryAllocatorAttachesTheTable() throws Exception {
    Map<String, String> expected = new TreeMap<>();
    NumpyTypes.NDARRAY_ATTRIBUTES.forEach(
        (name, type) -> expected.put(name, type.getName().toString()));
    List<String> mismatches = new ArrayList<>();
    int allocators = 0;
    for (String summary : SUMMARIES) {
      Document doc;
      try (InputStream in = getClass().getClassLoader().getResourceAsStream(summary)) {
        assertTrue("Missing summary " + summary + ".", in != null);
        doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(in);
      }
      NodeList methods = doc.getElementsByTagName("method");
      for (int m = 0; m < methods.getLength(); m++) {
        Element method = (Element) methods.item(m);
        Map<String, String> allocated = new TreeMap<>();
        NodeList news = method.getElementsByTagName("new");
        for (int n = 0; n < news.getLength(); n++) {
          Element nu = (Element) news.item(n);
          allocated.put(nu.getAttribute("def"), nu.getAttribute("class"));
        }
        for (Map.Entry<String, String> array : allocated.entrySet()) {
          if (!NDARRAY.equals(array.getValue())) continue;
          allocators++;
          Map<String, String> attached = new TreeMap<>();
          NodeList puts = method.getElementsByTagName("putfield");
          for (int p = 0; p < puts.getLength(); p++) {
            Element put = (Element) puts.item(p);
            if (!array.getKey().equals(put.getAttribute("ref"))) continue;
            String type = allocated.get(put.getAttribute("value"));
            if (type != null) attached.put(put.getAttribute("field"), type);
          }
          if (!expected.equals(attached))
            mismatches.add(
                summary
                    + " "
                    + ((Element) method.getParentNode()).getAttribute("name")
                    + "."
                    + method.getAttribute("name")
                    + " attaches "
                    + attached);
        }
      }
    }
    assertTrue("No array allocator found.", allocators > 0);
    assertEquals("Array allocators attaching other than " + expected + ".", List.of(), mismatches);
  }
}
