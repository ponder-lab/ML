package com.ibm.wala.cast.python.ipa.summaries;

import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.summaries.MethodSummary;
import com.ibm.wala.ssa.ConstantValue;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.util.collections.Pair;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParserFactory;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/**
 * Keyword arguments and {@code bool} constants in a summary file, which WALA's summary reader
 * cannot express (<a href="https://github.com/wala/ML/issues/1248">wala/ML#1248</a>, <a
 * href="https://github.com/wala/ML/issues/1120">wala/ML#1120</a>). A {@code <call>} element with a
 * {@code keywords} attribute passes its last arguments by the names it lists, in order, as a Python
 * call does, and a {@code <constant>} of {@code type="boolean"} is a Python {@code bool}. The
 * reader ignores both attributes, so this reads the file again and rewrites the summaries it made:
 * each such call becomes a {@link PythonInvokeInstruction} with those keywords, which the method
 * trampolines bind by name, and each such constant gets its value.
 */
public final class SummaryKeywordCalls {

  private static final Logger LOGGER = Logger.getLogger(SummaryKeywordCalls.class.getName());

  /** The constant types WALA's summary reader gives a value. */
  private static final Set<String> TYPED = Set.of("int", "long", "short", "float", "double");

  /** The attribute of a {@code <call>} naming the keywords its last arguments are passed by. */
  private static final String KEYWORDS = "keywords";

  /** What one summarized method declares beyond what the reader keeps. */
  private static final class Extras {

    /** The keywords of each call that has some, by the call's position among the method's calls. */
    private final Map<Integer, String[]> keywordCalls = new HashMap<>();

    /**
     * The value of each constant whose type the reader does not know, in the order the method
     * declares them: a {@code bool}'s value, or {@code null} for a type this does not know either.
     */
    private final List<Boolean> untypedConstants = new ArrayList<>();

    /** The number of the method's calls. */
    private int calls;
  }

  private SummaryKeywordCalls() {}

  /**
   * Rewrites the summaries read from a summary file for the keyword calls and {@code bool}
   * constants the file declares.
   *
   * @param xml The summary file, read again.
   * @param summaries The summaries WALA's reader made from the file, which this updates.
   */
  public static void rewrite(InputStream xml, Map<MethodReference, MethodSummary> summaries) {
    Map<String, Extras> extras = read(xml);
    if (extras.isEmpty()) return;

    for (Map.Entry<MethodReference, MethodSummary> entry : summaries.entrySet()) {
      MethodReference method = entry.getKey();
      Extras methodExtras = extras.get(key(method));
      if (methodExtras != null) entry.setValue(rewrite(entry.getValue(), methodExtras));
    }
  }

  private static String key(MethodReference method) {
    return method.getDeclaringClass().getName() + "." + method.getName();
  }

  private static MethodSummary rewrite(MethodSummary summary, Extras extras) {
    MethodSummary ret = new MethodSummary(summary.getMethod(), summary.getNumberOfParameters());
    ret.setStatic(summary.isStatic());
    ret.setFactory(summary.isFactory());
    if (summary.hasPoison()) {
      ret.addPoison(summary.getPoison());
      ret.setPoisonLevel(summary.getPoisonLevel());
    }
    ret.setValueNames(summary.getValueNames());
    Map<Integer, ConstantValue> constants =
        summary.getConstants() == null ? Map.of() : summary.getConstants();
    // The reader gives a constant of a type it does not know no value, and numbers constants in
    // the order the method declares them.
    List<Integer> untyped = new ArrayList<>();
    constants.forEach(
        (valueNumber, value) -> {
          if (value == null) untyped.add(valueNumber);
          else ret.addConstant(valueNumber, value);
        });
    Collections.sort(untyped);
    if (untyped.size() != extras.untypedConstants.size())
      throw new IllegalStateException(
          summary.getMethod()
              + " has "
              + untyped.size()
              + " constants of an unknown type, but its summary declares "
              + extras.untypedConstants.size());
    for (int i = 0; i < untyped.size(); i++) {
      Boolean value = extras.untypedConstants.get(i);
      ret.addConstant(untyped.get(i), value == null ? null : new ConstantValue(value));
    }

    int call = 0;
    for (SSAInstruction instruction : summary.getStatements()) {
      if (instruction == null) continue;
      if (instruction instanceof SSAAbstractInvokeInstruction invoke) {
        String[] keywords = extras.keywordCalls.get(call++);
        if (keywords != null) instruction = withKeywords(invoke, keywords);
      }
      ret.addStatement(instruction);
    }
    if (call != extras.calls)
      throw new IllegalStateException(
          summary.getMethod()
              + " has "
              + call
              + " calls, but its summary declares "
              + extras.calls);
    return ret;
  }

  private static PythonInvokeInstruction withKeywords(
      SSAAbstractInvokeInstruction invoke, String[] keywords) {
    int arguments = invoke.getNumberOfPositionalParameters();
    int positional = arguments - keywords.length;
    if (positional < 1)
      throw new IllegalStateException(invoke + " has fewer arguments than its keywords");
    int[] positionalParams = new int[positional];
    for (int i = 0; i < positional; i++) positionalParams[i] = invoke.getUse(i);
    @SuppressWarnings({"unchecked", "rawtypes"})
    Pair<String, Integer>[] keywordParams = new Pair[keywords.length];
    for (int i = 0; i < keywords.length; i++)
      keywordParams[i] = Pair.make(keywords[i], invoke.getUse(positional + i));
    int result = invoke.getNumberOfReturnValues() > 0 ? invoke.getReturnValue(0) : -1;
    LOGGER.fine(() -> "Passing " + String.join(", ", keywords) + " by keyword in " + invoke);
    return new PythonInvokeInstruction(
        invoke.iIndex(),
        result,
        invoke.getException(),
        invoke.getCallSite(),
        positionalParams,
        keywordParams);
  }

  private static Map<String, Extras> read(InputStream xml) {
    Map<String, Extras> ret = new HashMap<>();
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
                private Extras method;

                @Override
                public void startElement(
                    String uri, String localName, String qName, Attributes atts) {
                  switch (qName) {
                    case "package" -> packageName = atts.getValue("name");
                    case "class" -> className = atts.getValue("name");
                    case "method" -> method = new Extras();
                    case "call" -> {
                      if (method == null) return;
                      String keywords = atts.getValue(KEYWORDS);
                      if (keywords != null)
                        method.keywordCalls.put(method.calls, keywords.trim().split("\\s+"));
                      method.calls++;
                    }
                    case "constant" -> {
                      if (method != null && !TYPED.contains(atts.getValue("type")))
                        method.untypedConstants.add(
                            "boolean".equals(atts.getValue("type"))
                                ? Boolean.valueOf(atts.getValue("value"))
                                : null);
                    }
                    default -> {}
                  }
                  if (qName.equals("method"))
                    ret.put(
                        (packageName == null ? "L" : "L" + packageName + "/")
                            + className
                            + "."
                            + atts.getValue("name"),
                        method);
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
      throw new IllegalStateException("Cannot read a summary's keyword calls.", e);
    }
    ret.values()
        .removeIf(
            extras ->
                extras.keywordCalls.isEmpty()
                    && extras.untypedConstants.stream().allMatch(Objects::isNull));
    return ret;
  }
}
