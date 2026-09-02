package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.loader.PythonLoader.DynamicMethodBody;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.types.annotations.Annotation;
import java.io.File;
import java.util.Collections;
import java.util.List;
import org.junit.Test;

/**
 * Witnesses for wala/ML#868 layer one: a call-form decorator's argument names survive to the
 * loader's parallel channel ({@link DynamicMethodBody#getDecoratorCalls()}), while the name-only
 * annotation channel stays byte-identical to its prior form. Decoration is not applied in IR at all
 * (the decorator is never invoked and the raw function is bound), so this metadata is the only
 * place a decorator's arguments survive to; layer two's entrypoint modeling is its intended
 * consumer, and until that lands this channel is deliberately unread.
 *
 * <p>The decorated members pin the argument situations: {@code m} carries identifiers and an
 * explicit {@code None} (mined in order, {@code None} included, because a consumer binding
 * per-argument invocations needs the argument COUNT the program has); {@code n} carries a call
 * expression, which mines to the explicit unmineable marker rather than silently vanishing; {@code
 * p} is decorated bare, appearing with an empty argument list (the front end normalizes bare
 * application to a zero-argument call in the CAst) while its name-only annotation is unchanged,
 * which pins the no-behavior-change claim for everything that reads annotations today; and {@code
 * q} carries a dotted attribute-chain argument, the spelling real subjects use when classes are
 * referred to through an imported module.
 */
public class TestDecoratorCallArguments extends TestPythonMLCallGraphShape {

  @Test
  public void testDecoratorArgumentsSurviveToTheLoader() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "test_decorated_function_arguments.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    IClassHierarchy cha = builder.getClassHierarchy();

    DynamicMethodBody m = functionClass(cha, "m");
    assertEquals(
        "Expecting the decorator's identifier and None arguments mined in call order.",
        List.of(new Util.DecoratorCall("params", List.of("A", "B", "None"))),
        m.getDecoratorCalls());

    DynamicMethodBody n = functionClass(cha, "n");
    assertEquals(
        "Expecting a call-expression argument to surface as the unmineable marker, not to vanish:"
            + " a consumer must see the argument count the program has.",
        List.of(new Util.DecoratorCall("params", List.of(Util.UNMINEABLE_DECORATOR_ARGUMENT))),
        n.getDecoratorCalls());

    DynamicMethodBody p = functionClass(cha, "p");
    assertEquals(
        "Expecting a bare decorator to appear with an empty argument list: the front end"
            + " normalizes bare application to a zero-argument call in the CAst.",
        List.of(new Util.DecoratorCall("identity", List.of())),
        p.getDecoratorCalls());
    assertEquals(
        "Expecting the bare decorator's name-only annotation channel unchanged.",
        List.of(Annotation.make(TypeReference.findOrCreate(PythonTypes.pythonLoader, "Lidentity"))),
        List.copyOf(p.getAnnotations()));

    DynamicMethodBody q = functionClass(cha, "q");
    assertEquals(
        "Expecting a dotted (attribute-chain) argument mined through the OBJECT_REF chain: the"
            + " spelling real subjects use when classes are referred to through an imported"
            + " module.",
        List.of(new Util.DecoratorCall("params", List.of("Holder.C"))),
        q.getDecoratorCalls());
  }

  /**
   * Looks up the {@link DynamicMethodBody} for the fixture method of the given name on class {@code
   * T}.
   *
   * @param cha The class hierarchy of the analyzed fixture.
   * @param method The method's name.
   * @return Its function-object class.
   */
  private static DynamicMethodBody functionClass(IClassHierarchy cha, String method) {
    TypeReference reference =
        TypeReference.findOrCreate(
            PythonTypes.pythonLoader, "Lscript test_decorated_function_arguments.py/T/" + method);
    IClass klass = cha.lookupClass(reference);
    assertNotNull("Expecting the fixture method's function class: " + reference, klass);
    assertTrue(
        "Expecting a loader-defined function body carrying decorator metadata; got: "
            + klass.getClass().getName(),
        klass instanceof DynamicMethodBody);
    return (DynamicMethodBody) klass;
  }
}
