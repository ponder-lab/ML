package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import javax.xml.parsers.DocumentBuilderFactory;
import org.junit.Test;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * Every summary allocator of a dataset whose element is mirrored hands that element to each method
 * it installs that reads its own {@code mirrored_element} (wala/ML#1190). A method object reads the
 * element only from itself, so an allocator that installs the method without the hand-off leaves
 * the method's result unmirrored, and its members read as no tensor. The readers are found from the
 * summaries rather than listed, so a method that starts reading the element is checked at every
 * allocator, including allocators added on another branch.
 */
public class TestDatasetElementHandOffs {

  /** The summary that allocates datasets. */
  private static final String SUMMARY = "tensorflow.xml";

  /** The field a dataset, and each method object it installs, holds its mirrored element in. */
  private static final String MIRRORED_ELEMENT = "mirrored_element";

  /**
   * Each summary method that installs a reader of the mirrored element on a dataset whose element
   * it mirrors hands the reader that same element.
   *
   * @throws Exception On failure to read the summary.
   */
  @Test
  public void testEveryMirroredAllocatorHandsOffItsElement() throws Exception {
    Document doc;
    try (InputStream in = getClass().getClassLoader().getResourceAsStream(SUMMARY)) {
      assertTrue("Missing summary " + SUMMARY + ".", in != null);
      doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(in);
    }
    Set<String> readers = readers(doc);
    assertTrue("No method reads its mirrored element.", !readers.isEmpty());
    List<String> missing = new ArrayList<>();
    int installs = 0;
    NodeList methods = doc.getElementsByTagName("method");
    for (int m = 0; m < methods.getLength(); m++) {
      Element method = (Element) methods.item(m);
      Map<String, String> allocated = new HashMap<>();
      NodeList news = method.getElementsByTagName("new");
      for (int n = 0; n < news.getLength(); n++) {
        Element nu = (Element) news.item(n);
        allocated.put(nu.getAttribute("def"), nu.getAttribute("class"));
      }
      Map<String, String> mirrored = new HashMap<>();
      NodeList puts = method.getElementsByTagName("putfield");
      for (int p = 0; p < puts.getLength(); p++) {
        Element put = (Element) puts.item(p);
        if (MIRRORED_ELEMENT.equals(put.getAttribute("field")))
          mirrored.put(put.getAttribute("ref"), put.getAttribute("value"));
      }
      for (int p = 0; p < puts.getLength(); p++) {
        Element put = (Element) puts.item(p);
        String dataset = put.getAttribute("ref");
        String installed = put.getAttribute("value");
        String type = allocated.get(installed);
        if (MIRRORED_ELEMENT.equals(put.getAttribute("field"))
            || !mirrored.containsKey(dataset)
            || type == null
            || !readers.contains(type)) continue;
        installs++;
        if (!mirrored.get(dataset).equals(mirrored.get(installed)))
          missing.add(
              ((Element) method.getParentNode()).getAttribute("name")
                  + "."
                  + method.getAttribute("name")
                  + " installs "
                  + put.getAttribute("field")
                  + " without its element");
      }
    }
    assertTrue("No mirrored allocator installs a reader.", installs > 0);
    assertEquals("Readers installed without the mirrored element.", List.of(), missing);
  }

  /**
   * The classes, as a {@code <new>} names them, whose method reads its own mirrored element.
   *
   * @param doc The summary.
   * @return The classes' names.
   */
  private static Set<String> readers(Document doc) {
    Set<String> ret = new TreeSet<>();
    NodeList gets = doc.getElementsByTagName("getfield");
    for (int g = 0; g < gets.getLength(); g++) {
      Element get = (Element) gets.item(g);
      if (!MIRRORED_ELEMENT.equals(get.getAttribute("field"))
          || !"self".equals(get.getAttribute("ref"))) continue;
      Element type = (Element) get.getParentNode().getParentNode();
      StringBuilder name = new StringBuilder(type.getAttribute("name"));
      for (Node p = type.getParentNode(); p instanceof Element e; p = p.getParentNode())
        if ("package".equals(e.getTagName())) name.insert(0, e.getAttribute("name") + "/");
      ret.add("L" + name);
    }
    return ret;
  }
}
