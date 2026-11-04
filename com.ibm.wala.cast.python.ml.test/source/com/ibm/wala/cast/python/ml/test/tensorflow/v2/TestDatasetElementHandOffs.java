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
   * object reads from itself, exactly the values the summary method gives that field: the object's
   * own, written or read in the summary method, or else every value the summary method hands that
   * field to any installed method object reading it. A field takes several values where the summary
   * method writes it more than once, as {@code from_generator} hands each method both structures
   * its element may be declared by.
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
      for (Element method : children(type, "method"))
        for (Element statement : children(method, "getfield"))
          if ("self".equals(statement.getAttribute("ref")))
            readers
                .computeIfAbsent(statement.getAttribute("field"), k -> new TreeSet<>())
                .add(types.get(type));
    for (String field : DATASET_FIELDS)
      assertTrue("No method reads " + field + " from itself.", readers.containsKey(field));

    List<String> missing = new ArrayList<>();
    Map<String, Integer> checked = new TreeMap<>();
    for (Element type : classes)
      for (Element method : children(type, "method")) {
        String where = type.getAttribute("name") + "." + method.getAttribute("name");
        Map<String, String> allocated = new HashMap<>();
        Map<String, Map<String, Set<String>>> fields = new HashMap<>();
        for (Element statement : children(method, null)) {
          String ref = statement.getAttribute("ref");
          switch (statement.getTagName()) {
            case "new" ->
                allocated.put(statement.getAttribute("def"), statement.getAttribute("class"));
            case "putfield" -> value(fields, ref, statement).add(statement.getAttribute("value"));
            // An object's field read here holds what the object was given elsewhere, as a dataset's
            // element written by the summary that loads it.
            case "getfield" -> {
              if (!"self".equals(ref))
                value(fields, ref, statement).add(statement.getAttribute("def"));
            }
            default -> {}
          }
        }
        Map<String, Map<String, String>> installs = new LinkedHashMap<>();
        for (Element put : children(method, "putfield"))
          if (allocated.containsKey(put.getAttribute("value")))
            installs
                .computeIfAbsent(put.getAttribute("ref"), k -> new LinkedHashMap<>())
                .put(put.getAttribute("value"), put.getAttribute("field"));
        for (Map.Entry<String, Map<String, String>> object : installs.entrySet()) {
          Map<String, Set<String>> own = fields.getOrDefault(object.getKey(), Map.of());
          for (Map.Entry<String, Set<String>> field : readers.entrySet()) {
            List<String> installed = new ArrayList<>();
            for (String value : object.getValue().keySet())
              if (field.getValue().contains(allocated.get(value))) installed.add(value);
            Set<String> expected = new TreeSet<>(own.getOrDefault(field.getKey(), Set.of()));
            if (expected.isEmpty())
              for (String reader : installed)
                expected.addAll(handed(fields, reader, field.getKey()));
            if (expected.isEmpty()) continue;
            for (String reader : installed) {
              checked.merge(field.getKey(), 1, Integer::sum);
              Set<String> absent = new TreeSet<>(expected);
              absent.removeAll(handed(fields, reader, field.getKey()));
              Set<String> extra = new TreeSet<>(handed(fields, reader, field.getKey()));
              extra.removeAll(expected);
              if (!absent.isEmpty())
                missing.add(
                    where
                        + " installs "
                        + object.getValue().get(reader)
                        + " without its "
                        + field.getKey()
                        + " "
                        + absent);
              if (!extra.isEmpty())
                missing.add(
                    where
                        + " installs "
                        + object.getValue().get(reader)
                        + " with "
                        + field.getKey()
                        + " "
                        + extra
                        + " the object does not hold");
            }
          }
        }
      }
    for (String field : DATASET_FIELDS)
      assertTrue("No hand-off of " + field + " is checked.", checked.containsKey(field));
    assertEquals("Installed methods missing a field they read.", List.of(), missing);
  }

  /**
   * The values a summary method gives an object's field.
   *
   * @param fields Each object's fields' values.
   * @param object The object.
   * @param field The field.
   * @return The values, empty when none is given.
   */
  private static Set<String> handed(
      Map<String, Map<String, Set<String>>> fields, String object, String field) {
    return fields.getOrDefault(object, Map.of()).getOrDefault(field, Set.of());
  }

  /**
   * The values, to be added to, of the field a statement writes or reads on an object.
   *
   * @param fields Each object's fields' values.
   * @param object The object.
   * @param statement The statement.
   * @return The field's values.
   */
  private static Set<String> value(
      Map<String, Map<String, Set<String>>> fields, String object, Element statement) {
    return fields
        .computeIfAbsent(object, k -> new HashMap<>())
        .computeIfAbsent(statement.getAttribute("field"), k -> new TreeSet<>());
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
   * The child elements of an element, as a method's statements or a class's methods.
   *
   * @param element A class or a method.
   * @param tag The children's tag, or {@code null} for every child.
   * @return The children, in document order.
   */
  private static List<Element> children(Element element, String tag) {
    List<Element> ret = new ArrayList<>();
    for (Node n = element.getFirstChild(); n != null; n = n.getNextSibling())
      if (n instanceof Element child && (tag == null || tag.equals(child.getTagName())))
        ret.add(child);
    return ret;
  }
}
