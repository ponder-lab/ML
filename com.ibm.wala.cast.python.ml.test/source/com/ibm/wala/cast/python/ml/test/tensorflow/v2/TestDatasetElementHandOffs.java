package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import javax.xml.parsers.DocumentBuilderFactory;
import org.junit.Test;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * Every summary method that installs method objects on an object hands each of them the fields it
 * reads from itself, with the value the object holds (wala/ML#1190). A dataset's methods read its
 * element, its marker that the element is not mirrored, and its element's structure only from
 * themselves, so an allocator that installs a method without one of those hand-offs changes what
 * the method's result reads: a missing element leaves the result unmirrored and its members no
 * tensor, and a missing marker makes a result read as mirrored when its receiver is not.
 *
 * <p>The fields each method reads are found from the summaries rather than listed, so a method that
 * starts reading a field is checked at every allocator that installs it, including allocators added
 * on another branch.
 */
public class TestDatasetElementHandOffs {

  /** The summary files the engine loads. */
  private static final List<String> SUMMARIES =
      List.of(
          "numpy.xml",
          "scipy.xml",
          "sentencepiece.xml",
          "sklearn.xml",
          "tensorflow.xml",
          "tensorflow_addons.xml",
          "tensorflow_datasets.xml");

  /**
   * The fields of a dataset its methods read, which the check must find: a rename that hid one from
   * it would otherwise pass.
   */
  private static final List<String> DATASET_FIELDS =
      List.of("mirrored_element", "legacy_element", "element_structure");

  /**
   * Each method object a summary method installs on an object receives, for every field the method
   * object reads from itself, the value the summary method gives that field: the object's own,
   * written or read in the summary method, or else the one value the summary method hands that
   * field to every installed method object reading it.
   *
   * @throws Exception On failure to read a summary.
   */
  @Test
  public void testEveryInstalledMethodReceivesTheFieldsItReads() throws Exception {
    List<Element> classes = new ArrayList<>();
    Map<Element, String> types = new HashMap<>();
    for (String summary : SUMMARIES) {
      Document doc;
      try (InputStream in = getClass().getClassLoader().getResourceAsStream(summary)) {
        assertTrue("Missing summary " + summary + ".", in != null);
        doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(in);
      }
      NodeList list = doc.getElementsByTagName("class");
      for (int c = 0; c < list.getLength(); c++) {
        Element type = (Element) list.item(c);
        classes.add(type);
        types.put(type, typeName(type));
      }
    }
    Map<String, Set<String>> readers = new TreeMap<>();
    for (Element type : classes)
      for (Element statement : statements(type, "getfield"))
        if ("self".equals(statement.getAttribute("ref")))
          readers
              .computeIfAbsent(statement.getAttribute("field"), k -> new TreeSet<>())
              .add(types.get(type));
    for (String field : DATASET_FIELDS)
      assertTrue("No method reads " + field + " from itself.", readers.containsKey(field));

    List<String> missing = new ArrayList<>();
    Map<String, Integer> checked = new TreeMap<>();
    for (Element type : classes) {
      NodeList methods = type.getElementsByTagName("method");
      for (int m = 0; m < methods.getLength(); m++) {
        Element method = (Element) methods.item(m);
        String where = type.getAttribute("name") + "." + method.getAttribute("name");
        Map<String, String> allocated = new HashMap<>();
        Map<String, Map<String, String>> fields = new HashMap<>();
        Map<String, Map<String, String>> installs = new LinkedHashMap<>();
        for (Node n = method.getFirstChild(); n != null; n = n.getNextSibling()) {
          if (!(n instanceof Element statement)) continue;
          String ref = statement.getAttribute("ref");
          switch (statement.getTagName()) {
            case "new" ->
                allocated.put(statement.getAttribute("def"), statement.getAttribute("class"));
            case "putfield" ->
                fields
                    .computeIfAbsent(ref, k -> new HashMap<>())
                    .put(statement.getAttribute("field"), statement.getAttribute("value"));
            // An object's field read here holds what the object was given elsewhere, as a dataset's
            // element written by the summary that loads it.
            case "getfield" -> {
              if (!"self".equals(ref))
                fields
                    .computeIfAbsent(ref, k -> new HashMap<>())
                    .put(statement.getAttribute("field"), statement.getAttribute("def"));
            }
            default -> {}
          }
        }
        for (Element put : statements(method, "putfield"))
          if (allocated.containsKey(put.getAttribute("value")))
            installs
                .computeIfAbsent(put.getAttribute("ref"), k -> new LinkedHashMap<>())
                .put(put.getAttribute("value"), put.getAttribute("field"));
        for (Map.Entry<String, Map<String, String>> object : installs.entrySet()) {
          Map<String, String> own = fields.getOrDefault(object.getKey(), Map.of());
          for (Map.Entry<String, Set<String>> field : readers.entrySet()) {
            List<String> installed = new ArrayList<>();
            for (String value : object.getValue().keySet())
              if (field.getValue().contains(allocated.get(value))) installed.add(value);
            if (installed.isEmpty()) continue;
            Set<String> expected = new TreeSet<>();
            if (own.containsKey(field.getKey())) expected.add(own.get(field.getKey()));
            else
              for (String value : installed) {
                String handed = fields.getOrDefault(value, Map.of()).get(field.getKey());
                if (handed != null) expected.add(handed);
              }
            if (expected.isEmpty()) continue;
            if (expected.size() > 1) {
              missing.add(where + " hands " + field.getKey() + " as each of " + expected);
              continue;
            }
            String value = expected.iterator().next();
            for (String reader : installed) {
              checked.merge(field.getKey(), 1, Integer::sum);
              if (!value.equals(fields.getOrDefault(reader, Map.of()).get(field.getKey())))
                missing.add(
                    where
                        + " installs "
                        + object.getValue().get(reader)
                        + " without its "
                        + field.getKey());
            }
          }
        }
      }
    }
    for (String field : DATASET_FIELDS)
      assertTrue("No hand-off of " + field + " is checked.", checked.containsKey(field));
    assertEquals("Installed methods missing a field they read.", List.of(), missing);
  }

  /**
   * A class's name as a {@code <new>} writes it.
   *
   * @param type The class.
   * @return {@code L} followed by its packages' names and its own, separated by slashes.
   */
  private static String typeName(Element type) {
    StringBuilder name = new StringBuilder(type.getAttribute("name"));
    for (Node p = type.getParentNode(); p instanceof Element e; p = p.getParentNode())
      if ("package".equals(e.getTagName())) name.insert(0, e.getAttribute("name") + "/");
    return "L" + name;
  }

  /**
   * The statements of a tag within an element.
   *
   * @param element A class or a method.
   * @param tag The statement's tag.
   * @return The statements, in document order.
   */
  private static List<Element> statements(Element element, String tag) {
    List<Element> ret = new ArrayList<>();
    NodeList list = element.getElementsByTagName(tag);
    for (int i = 0; i < list.getLength(); i++) ret.add((Element) list.item(i));
    return ret;
  }
}
