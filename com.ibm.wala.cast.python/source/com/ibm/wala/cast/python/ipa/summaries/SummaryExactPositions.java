package com.ibm.wala.cast.python.ipa.summaries;

import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.Set;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParserFactory;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/**
 * The lists and tuples a summary file says it allocates with exact run-time positions (<a
 * href="https://github.com/wala/ML/issues/1265">wala/ML#1265</a>). A {@code <new>} element with
 * {@code positions="exact"} allocates a container whose constant indices the summary writes are the
 * container's run-time positions, so the container query reports its length and the types at its
 * positions as it does for a display the program allocates. A summary that says nothing keeps the
 * query's default for a summary's allocation: its slots are placeholders, and it reports none.
 * WALA's summary reader ignores the attribute, so this reads the file again, as {@link
 * SummaryKeywordCalls} does.
 */
public final class SummaryExactPositions {

  /** The attribute of a {@code <new>} saying how its container's positions relate to run time. */
  private static final String POSITIONS = "positions";

  /** The value of {@link #POSITIONS} saying the summary's positions are the run-time ones. */
  private static final String EXACT = "exact";

  /**
   * A summary's allocation of a container with exact positions.
   *
   * @param method The summarized method, as its class's name and its own, {@code Lpkg/cls.name}.
   * @param allocated The name of the class it allocates.
   * @param ordinal Its place among the method's allocations of that class, from {@code 0}.
   */
  public record Allocation(String method, String allocated, int ordinal) {}

  private SummaryExactPositions() {}

  /**
   * Reads the allocations a summary file marks as having exact positions.
   *
   * @param xml The summary file, which this closes.
   * @return The marked allocations.
   * @throws IllegalStateException If the file cannot be read.
   */
  public static Set<Allocation> read(InputStream xml) {
    Set<Allocation> ret = HashSetFactory.make();
    try (xml) {
      SAXParserFactory factory = SAXParserFactory.newInstance();
      factory.setFeature("http://apache.org/xml/features/nonvalidating/load-external-dtd", false);
      factory.setFeature("http://xml.org/sax/features/external-general-entities", false);
      factory.setFeature("http://xml.org/sax/features/external-parameter-entities", false);
      factory
          .newSAXParser()
          .parse(
              xml,
              new DefaultHandler() {
                private String packageName;
                private String className;
                private String method;

                /** The method's allocations so far, by the class each allocates. */
                private final Map<String, Integer> allocations = HashMapFactory.make();

                @Override
                public void startElement(
                    String uri, String localName, String qName, Attributes atts) {
                  switch (qName) {
                    case "package" -> packageName = atts.getValue("name");
                    case "class" -> className = atts.getValue("name");
                    case "method" -> {
                      method =
                          (packageName == null ? "L" : "L" + packageName + "/")
                              + className
                              + "."
                              + atts.getValue("name");
                      allocations.clear();
                    }
                    case "new" -> {
                      if (method == null) return;
                      String allocated = atts.getValue("class");
                      int ordinal = allocations.merge(allocated, 1, Integer::sum) - 1;
                      if (EXACT.equals(atts.getValue(POSITIONS)))
                        ret.add(new Allocation(method, allocated, ordinal));
                    }
                    default -> {}
                  }
                }

                @Override
                public void endElement(String uri, String localName, String qName) {
                  switch (qName) {
                    case "package" -> packageName = null;
                    case "class" -> className = null;
                    case "method" -> method = null;
                    default -> {}
                  }
                }
              });
    } catch (IOException | ParserConfigurationException | SAXException e) {
      throw new IllegalStateException("Cannot read a summary's exact-position allocations.", e);
    }
    return ret;
  }

  /**
   * Whether an allocation site of a node is one of the given marked allocations. A summarized
   * method other than {@code do} runs as the {@code do} of a function class named for it (see
   * {@code PythonAnalysisEngine.addSummaryBypassLogic}), so the node's method is matched under both
   * names.
   *
   * @param marked The marked allocations.
   * @param node The node allocating.
   * @param site The allocation site.
   * @return {@code true} iff the site is marked.
   */
  public static boolean allocates(Set<Allocation> marked, CGNode node, NewSiteReference site) {
    if (marked.isEmpty()) return false;
    IR ir = node.getIR();
    if (ir == null) return false;
    String allocated = site.getDeclaredType().getName().toString();
    int ordinal = 0;
    boolean found = false;
    for (SSAInstruction instruction : ir.getInstructions())
      if (instruction instanceof SSANewInstruction allocation
          && allocation.getNewSite().getDeclaredType().getName().toString().equals(allocated)) {
        if (allocation.getNewSite().equals(site)) {
          found = true;
          break;
        }
        ordinal++;
      }
    if (!found) return false;
    IMethod method = node.getMethod();
    String declaring = method.getDeclaringClass().getName().toString();
    if (marked.contains(new Allocation(declaring + "." + method.getName(), allocated, ordinal)))
      return true;
    int slash = declaring.lastIndexOf('/');
    return slash > 0
        && marked.contains(
            new Allocation(
                declaring.substring(0, slash) + "." + declaring.substring(slash + 1),
                allocated,
                ordinal));
  }
}
