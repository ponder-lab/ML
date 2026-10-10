package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.FileSystem;
import java.nio.file.FileSystemAlreadyExistsException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.Test;

/**
 * Tests that a relative import names a module of the importer's own package, whichever directory
 * the project is checked out under and in whichever order its modules are given (<a
 * href="https://github.com/wala/ML/issues/168">wala/ML#168</a>). The fixture's {@code pkg/layers}
 * re-exports {@code Layer} by {@code from .conv.dense import Layer}, while the sibling package
 * {@code pkg/nn} holds a module of the same name, {@code conv/dense.py}, and re-exports from it the
 * same way. The relative import was resolved to its last component alone, {@code dense}, and the
 * module chosen among the in-scope scripts ending in {@code dense.py} by the iteration order of a
 * map keyed by their absolute paths, so the checkout's directory name decided which module each
 * re-export bound.
 */
public class TestRelativeImportResolution extends TestPythonMLCallGraphShape {

  private static final String FIXTURE = "relimport_proj";

  /** Directory names to check the project out under; each spells a different absolute path. */
  private static final List<String> NAMES =
      List.of(
          "a",
          "b",
          "cc",
          "dd",
          "eee",
          "ffff",
          "g7",
          "h8",
          "checkout",
          "copy-2",
          "x",
          "yy",
          "zzz",
          "project",
          "work",
          "tmp-q");

  /** What an analysis of a checkout yields, keyed so two checkouts can be compared. */
  private record Analysis(
      Map<String, Integer> nodes, Set<String> edges, Map<String, Set<TensorType>> types) {}

  /**
   * Copies the fixture under a directory of the given name.
   *
   * @param parent The directory to copy under.
   * @param name The checkout's directory name.
   * @return The copy's project root.
   */
  private static Path checkOut(Path parent, String name) throws IOException, URISyntaxException {
    URL resource = TestRelativeImportResolution.class.getResource("/" + FIXTURE);
    assertTrue("The fixture is on the test class path", resource != null);
    URI uri = resource.toURI();
    Path target = parent.resolve(name).resolve(FIXTURE);
    // The fixture is a directory when the test module's classes are, and an entry of its jar when
    // the reactor resolves the module as a packaged artifact, as a full build does.
    if ("jar".equals(uri.getScheme())) {
      FileSystem jar;
      try {
        jar = FileSystems.newFileSystem(uri, Map.of());
      } catch (FileSystemAlreadyExistsException e) {
        jar = FileSystems.getFileSystem(uri);
      }
      copyTree(jar.getPath("/" + FIXTURE), target);
    } else copyTree(Path.of(uri), target);
    return target;
  }

  /**
   * Copies a tree's Python files, and its directories, under a target directory.
   *
   * @param source The tree's root, on any file system.
   * @param target The directory to copy it under.
   */
  private static void copyTree(Path source, Path target) throws IOException {
    try (Stream<Path> walk = Files.walk(source)) {
      for (Path p : walk.sorted().collect(Collectors.toList())) {
        Path to = target.resolve(source.relativize(p).toString());
        if (Files.isDirectory(p)) Files.createDirectories(to);
        else if (p.toString().endsWith(".py")) Files.copy(p, to);
      }
    }
  }

  /**
   * Analyzes a checkout, handing the engine its modules in the given order.
   *
   * @param root The checkout's project root, its PYTHONPATH entry.
   * @param reversed Whether to give the modules in reverse path order.
   * @return The call graph's nodes and edges and the tensor types, by method signature.
   */
  private Analysis analyze(Path root, boolean reversed)
      throws IOException, ClassHierarchyException, CancelException {
    List<Path> files;
    try (Stream<Path> walk = Files.walk(root)) {
      files = walk.filter(p -> p.toString().endsWith(".py")).sorted().collect(Collectors.toList());
    }
    if (reversed) Collections.reverse(files);
    // An ordered list: the harness's makeEngine collects modules into a hash set, which loses it.
    List<Module> modules = new ArrayList<>();
    for (Path f : files) modules.add(getScript(f.toString()));
    PythonTensorAnalysisEngine engine =
        new PythonTensorAnalysisEngine(
            List.of(root.toFile()),
            PythonTensorAnalysisEngine.TENSORFLOW,
            PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH);
    engine.setModuleFiles(modules);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    Map<String, Integer> nodes = new TreeMap<>();
    Set<String> edges = new TreeSet<>();
    for (CGNode n : cg) {
      String from = n.getMethod().getSignature();
      nodes.merge(from, 1, Integer::sum);
      for (CGNode s : (Iterable<CGNode>) () -> cg.getSuccNodes(n))
        edges.add(from + " -> " + s.getMethod().getSignature());
    }
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);
    Map<String, Set<TensorType>> types = new TreeMap<>();
    analysis.forEach(
        p -> {
          if (p.fst instanceof LocalPointerKey key && p.snd.getTypes() != null)
            types
                .computeIfAbsent(
                    key.getNode().getMethod().getSignature() + "#" + key.getValueNumber(),
                    k -> new HashSet<>())
                .addAll(p.snd.getTypes());
        });
    return new Analysis(nodes, edges, types);
  }

  private static Set<TensorType> parameterTypes(Analysis a, String function) {
    return a.types().getOrDefault("script main.py." + function + ".do()LRoot;#2", Set.of());
  }

  private static void delete(Path dir) throws IOException {
    try (Stream<Path> walk = Files.walk(dir)) {
      for (Path p : walk.sorted(Comparator.reverseOrder()).collect(Collectors.toList()))
        Files.delete(p);
    }
  }

  /**
   * Each re-export binds the module of its own package under every directory name: the layer's
   * constructor is a node, and the two consumers read the layer's {@code (2, 3)} product and the
   * sibling package's {@code (5,)} tensor.
   */
  @Test
  public void testRelativeImportBindsImportersPackage() throws Exception {
    Path parent = Files.createTempDirectory("relimport");
    try {
      List<String> wrong = new ArrayList<>();
      for (String name : NAMES) {
        Analysis a = analyze(checkOut(parent, name), false);
        boolean layer =
            a.nodes().keySet().stream()
                .anyMatch(s -> s.contains("pkg.layers.conv.dense.py.Layer.__init__."));
        Set<TensorType> scaled = parameterTypes(a, "consume_scaled");
        Set<TensorType> dense = parameterTypes(a, "consume_dense");
        if (!layer
            || !scaled.equals(Set.of(TensorType.of(DType.FLOAT32, 2, 3)))
            || !dense.equals(Set.of(TensorType.of(DType.FLOAT32, 5))))
          wrong.add(
              name + " (Layer.__init__ " + layer + ", scaled " + scaled + ", dense " + dense + ")");
      }
      assertEquals("Checkouts binding a re-export to the wrong module", List.of(), wrong);
    } finally {
      delete(parent);
    }
  }

  /**
   * The call graph's nodes and edges and the tensor types are the same under every directory name
   * and in either module order.
   */
  @Test
  public void testDirectoryNameAndOrderInvariance() throws Exception {
    Path parent = Files.createTempDirectory("relimport");
    try {
      Analysis first = null;
      String firstLabel = null;
      for (String name : NAMES.subList(0, 6))
        for (boolean reversed : new boolean[] {false, true}) {
          Analysis a = analyze(checkOut(parent, name + (reversed ? "-r" : "")), reversed);
          String label = name + (reversed ? " reversed" : " sorted");
          if (first == null) {
            first = a;
            firstLabel = label;
            continue;
          }
          assertEquals(label + " nodes vs " + firstLabel, first.nodes(), a.nodes());
          assertEquals(label + " edges vs " + firstLabel, first.edges(), a.edges());
          assertEquals(label + " tensor types vs " + firstLabel, first.types(), a.types());
        }
    } finally {
      delete(parent);
    }
  }
}
