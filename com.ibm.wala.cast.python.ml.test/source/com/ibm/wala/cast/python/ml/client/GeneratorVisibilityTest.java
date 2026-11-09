package com.ibm.wala.cast.python.ml.client;

import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.constructors;
import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.methods;
import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noMethods;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.tngtech.archunit.core.domain.JavaClass;
import com.tngtech.archunit.core.domain.JavaClasses;
import com.tngtech.archunit.core.importer.ClassFileImporter;
import com.tngtech.archunit.core.importer.ImportOption;
import com.tngtech.archunit.lang.ArchRule;
import java.util.Set;
import org.junit.Test;

/**
 * A client may subclass a generator but neither build one nor read its tensor types outside the
 * analysis (<a href="https://github.com/wala/ML/issues/1191">wala/ML#1191</a>). These rules keep a
 * generator added later from reopening either.
 */
public class GeneratorVisibilityTest {

  /**
   * The test module, whose classes share the generators' package. It compiles its tests as main
   * classes, so {@link ImportOption.Predefined#DO_NOT_INCLUDE_TESTS} keeps them.
   */
  private static final String TEST_MODULE = "com.ibm.wala.cast.python.ml.test";

  /** The analysis's classes in the generators' package, without the test module's. */
  private static final JavaClasses CLIENT =
      new ClassFileImporter()
          .withImportOption(location -> !location.contains(TEST_MODULE))
          .importPackages(TensorGenerator.class.getPackageName());

  /** A public generator's constructors are not public, so only a subclass calls them. */
  private static final ArchRule CONSTRUCTORS =
      constructors()
          .that()
          .areDeclaredInClassesThat()
          .arePublic()
          .and()
          .areDeclaredInClassesThat()
          .areAssignableTo(TensorGenerator.class)
          .should()
          .notBePublic()
          .because("a client may subclass a generator but not build one (wala/ML#1191)");

  /**
   * A public generator's one-argument tensor-type reads, its legacy, record and default reads, are
   * not public.
   */
  private static final ArchRule READS =
      methods()
          .that()
          .areDeclaredInClassesThat()
          .arePublic()
          .and()
          .areDeclaredInClassesThat()
          .areAssignableTo(TensorGenerator.class)
          .and()
          .haveNameMatching(
              "getTensorTypes|getShapes|getDTypes|getShapeResult|getDefaultShapes|getDefaultDTypes"
                  + "|getDefaultShapeResult")
          .and()
          .haveRawParameterTypes(PropagationCallGraphBuilder.class)
          .should()
          .notBePublic()
          .because("a client reads tensor types only through the analysis (wala/ML#1191)");

  /**
   * No public static method of a public class hands out a generator, as the factory's dispatch
   * would if it were public.
   */
  private static final ArchRule FACTORIES =
      noMethods()
          .that()
          .areDeclaredInClassesThat()
          .arePublic()
          .and()
          .arePublic()
          .and()
          .areStatic()
          .should()
          .haveRawReturnType(JavaClass.Predicates.assignableTo(TensorGenerator.class))
          .because("a client reaches a generator only through its own subclass (wala/ML#1191)");

  /** A generator that a client could build, directly or through a factory, as the rules forbid. */
  public static class PublicGenerator extends LogSoftmax {

    public PublicGenerator(PointsToSetVariable source) {
      super(source);
    }

    public static PublicGenerator of(PointsToSetVariable source) {
      return new PublicGenerator(source);
    }
  }

  /** A generator whose tensor-type read a client could call, as the rules forbid. */
  public static class PublicReadGenerator extends LogSoftmax {

    protected PublicReadGenerator(PointsToSetVariable source) {
      super(source);
    }

    @Override
    public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
      return super.getTensorTypes(builder);
    }
  }

  /** The import holds the generators, so the rules below are not checked against nothing. */
  @Test
  public void testImportHoldsGenerators() {
    assertTrue(CLIENT.contain(TensorGenerator.class));
    assertTrue(CLIENT.contain(LogSoftmax.class));
  }

  /** No generator has a public constructor. */
  @Test
  public void testConstructorsAreNotPublic() {
    CONSTRUCTORS.check(CLIENT);
  }

  /** No generator has a public one-argument tensor-type read. */
  @Test
  public void testReadsAreNotPublic() {
    READS.check(CLIENT);
  }

  /** No public static method hands out a generator. */
  @Test
  public void testNoPublicFactories() {
    FACTORIES.check(CLIENT);
  }

  /** Each rule reports a generator that breaks it. */
  @Test
  public void testRulesCatchPublicMembers() {
    JavaClasses witnesses =
        new ClassFileImporter().importClasses(PublicGenerator.class, PublicReadGenerator.class);
    assertTrue(CONSTRUCTORS.evaluate(witnesses).hasViolation());
    assertTrue(READS.evaluate(witnesses).hasViolation());
    assertTrue(FACTORIES.evaluate(witnesses).hasViolation());
  }
}
