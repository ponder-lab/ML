package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory.Dispatch;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests which dispatches a {@link DispatchPath} keeps for its walk and which it hands the factory
 * to memoize per builder, over graphs of sources walked as {@link TensorGeneratorFactory} walks
 * them (<a href="https://github.com/wala/ML/issues/1274">wala/ML#1274</a>).
 */
public class TestDispatchPath {

  /** A graph of sources, and a walk of it that counts the dispatches computed. */
  private static final class Walk {

    /** The sources, by name. */
    private final Map<String, PointsToSetVariable> sources = new HashMap<>();

    /** Each source's successors, in the order its dispatch reaches them. */
    private final Map<String, List<String>> successors = new HashMap<>();

    /** How many times each source's dispatch was computed. */
    private final Map<String, Integer> computed = new HashMap<>();

    /** The sources whose dispatches the path handed over to memoize per builder. */
    private final Map<String, Dispatch> memoized = new HashMap<>();

    /** The sources the dispatches were cut at. */
    private final Set<String> cuts = new HashSet<>();

    /**
     * Adds a source.
     *
     * @param name Its name.
     * @param successors The sources its dispatch reaches, in order.
     * @return This walk.
     */
    private Walk source(String name, String... successors) {
      this.sources.put(name, new PointsToSetVariable(new PointerKey() {}));
      this.successors.put(name, List.of(successors));
      return this;
    }

    /**
     * Dispatches a source as the factory does: a source on the path is cut, a memoized or kept
     * dispatch is replayed, and any other is computed by dispatching its successors.
     *
     * @param path The path.
     * @param name The source's name.
     */
    private void dispatch(DispatchPath path, String name) {
      PointsToSetVariable source = this.sources.get(name);
      if (path.cuts(source)) {
        this.cuts.add(name);
        return;
      }
      Dispatch memoized = this.memoized.get(name);
      if (memoized != null) {
        path.replays(memoized);
        return;
      }
      if (path.kept(source) != null) return;
      this.computed.merge(name, 1, Integer::sum);
      path.enter(source);
      for (String successor : this.successors.get(name)) this.dispatch(path, successor);
      Dispatch dispatch = path.leave(source, null, null);
      if (dispatch != null) this.memoized.put(name, dispatch);
    }

    /**
     * How many times a source's dispatch was computed.
     *
     * @param name The source's name.
     * @return The count.
     */
    private int computed(String name) {
      return this.computed.getOrDefault(name, 0);
    }
  }

  /**
   * A diamond with no cycle: the shared source is computed once, and every dispatch is handed over
   * to memoize, each one exact.
   */
  @Test
  public void testDiamond() {
    Walk walk =
        new Walk()
            .source("top", "left", "right")
            .source("left", "shared")
            .source("right", "shared")
            .source("shared");
    walk.dispatch(new DispatchPath(), "top");
    assertEquals(1, walk.computed("shared"));
    assertEquals(Set.of("top", "left", "right", "shared"), walk.memoized.keySet());
    for (Dispatch dispatch : walk.memoized.values()) assertTrue(dispatch.exact());
    assertTrue(walk.cuts.isEmpty());
  }

  /**
   * The same diamond with its shared source reaching the top: the walk still computes the shared
   * source once, replaying its dispatch at the second visit, and hands over only the dispatch of
   * the source it entered the cycle at, which is not exact.
   */
  @Test
  public void testDiamondInCycle() {
    Walk walk =
        new Walk()
            .source("top", "left", "right")
            .source("left", "shared")
            .source("right", "shared")
            .source("shared", "top");
    DispatchPath path = new DispatchPath();
    long reached = DispatchPath.reachedOnThread();
    walk.dispatch(path, "top");
    assertEquals(1, walk.computed("shared"));
    assertEquals(1, walk.computed("left"));
    assertEquals(1, walk.computed("right"));
    assertEquals(Set.of("top"), walk.cuts);
    assertEquals(Set.of("top"), walk.memoized.keySet());
    assertFalse(walk.memoized.get("top").exact());
    // The cut, and the replay of the shared source's dispatch, which depended on the cut.
    assertEquals(reached + 2, DispatchPath.reachedOnThread());
    // The cycle's other dispatches held while its first source was on the path.
    assertNull(path.kept(walk.sources.get("shared")));
    assertNull(path.kept(walk.sources.get("left")));
    assertNotNull(path.kept(walk.sources.get("top")));
  }

  /**
   * A source reached again after its cycle is complete is computed again, replaying the memoized
   * dispatch of the cycle's first source, and its dispatch is then handed over too. It was not cut,
   * and is not exact either: the dispatch it replayed was cut.
   */
  @Test
  public void testCycleMemberAfterItsCycle() {
    Walk walk = new Walk().source("first", "second").source("second", "third").source("third");
    walk.successors.put("third", List.of("first"));
    walk.dispatch(new DispatchPath(), "first");
    assertEquals(Set.of("first"), walk.memoized.keySet());
    walk.dispatch(new DispatchPath(), "second");
    assertEquals(2, walk.computed("second"));
    assertEquals(2, walk.computed("third"));
    assertEquals(1, walk.computed("first"));
    assertEquals(Set.of("first", "second", "third"), walk.memoized.keySet());
    assertTrue(walk.cuts.equals(Set.of("first")));
    for (Dispatch dispatch : walk.memoized.values()) assertFalse(dispatch.exact());
  }

  /**
   * A chain of diamonds whose last source reaches its first: the walk computes each source once,
   * where one computing a source again at every visit doubles its count at each diamond.
   */
  @Test
  public void testChainOfDiamondsInCycle() {
    int diamonds = 20;
    Walk walk = new Walk();
    List<String> names = new ArrayList<>();
    for (int i = 0; i < diamonds; i++) {
      walk.source("top" + i, "left" + i, "right" + i);
      walk.source("left" + i, "top" + (i + 1));
      walk.source("right" + i, "top" + (i + 1));
      names.addAll(List.of("top" + i, "left" + i, "right" + i));
    }
    walk.source("top" + diamonds, "top0");
    names.add("top" + diamonds);
    walk.dispatch(new DispatchPath(), "top0");
    for (String name : names) assertEquals(name, 1, walk.computed(name));
    assertEquals(Set.of("top0"), walk.memoized.keySet());
  }

  /**
   * Two cycles, the second reached from the first: the second's first source is handed over when
   * its cycle completes, though the walk is still inside the first cycle, and the first cycle's
   * dispatches replay it.
   */
  @Test
  public void testNestedCycles() {
    Walk walk =
        new Walk()
            .source("outer", "middle")
            .source("middle", "inner", "outer")
            .source("inner", "leaf", "inner")
            .source("leaf");
    DispatchPath path = new DispatchPath();
    walk.dispatch(path, "outer");
    assertEquals(Set.of("outer", "inner", "leaf"), walk.memoized.keySet());
    assertTrue(walk.memoized.get("leaf").exact());
    assertFalse(walk.memoized.get("inner").exact());
    assertFalse(walk.memoized.get("outer").exact());
  }

  /**
   * A dispatch reaching a source that another path of its thread is resolving is kept for its walk,
   * so the walk computes it once, and is not handed over; nor is any dispatch above it.
   */
  @Test
  public void testSourceOfAnotherPath() {
    Walk walk =
        new Walk()
            .source("outside")
            .source("top", "left", "right")
            .source("left", "shared")
            .source("right", "shared")
            .source("shared", "outside")
            .source("apart");
    DispatchPath outer = new DispatchPath();
    outer.enter(walk.sources.get("outside"));
    try {
      DispatchPath path = new DispatchPath();
      walk.dispatch(path, "top");
      assertEquals(1, walk.computed("shared"));
      // The other path's source is resolved by this walk, not cut.
      assertEquals(1, walk.computed("outside"));
      assertTrue(walk.cuts.isEmpty());
      assertEquals(Set.of("outside"), walk.memoized.keySet());
      walk.dispatch(path, "apart");
      assertTrue(walk.memoized.containsKey("apart"));
    } finally {
      outer.abandon(walk.sources.get("outside"));
    }
  }

  /**
   * A path that does not memoize keeps nothing and hands nothing over, so its walk computes a
   * source at every visit.
   */
  @Test
  public void testUnmemoizedPath() {
    Walk walk =
        new Walk()
            .source("top", "left", "right")
            .source("left", "shared")
            .source("right", "shared")
            .source("shared", "top");
    walk.dispatch(new DispatchPath(false), "top");
    assertEquals(2, walk.computed("shared"));
    assertTrue(walk.memoized.isEmpty());
  }

  /**
   * A dispatch that replays the dispatch of a cycle's first source after the cycle is complete
   * depends on no source on the path, so it is handed over, and so is the dispatch above it.
   * Neither is exact, the replayed dispatch having been cut.
   */
  @Test
  public void testReplayOfCompletedCycle() {
    Walk walk = new Walk().source("top").source("first").source("after");
    PointsToSetVariable top = walk.sources.get("top");
    PointsToSetVariable first = walk.sources.get("first");
    PointsToSetVariable after = walk.sources.get("after");
    DispatchPath path = new DispatchPath();
    long reached = DispatchPath.reachedOnThread();
    path.enter(top);
    path.enter(first);
    assertTrue(path.cuts(first));
    assertNotNull(path.leave(first, null, null));
    path.enter(after);
    assertNotNull(path.kept(first));
    // The replayed dispatch was cut at its own source alone, which reaches no source being
    // resolved.
    assertEquals(reached + 1, DispatchPath.reachedOnThread());
    Dispatch replaying = path.leave(after, null, null);
    assertNotNull(replaying);
    assertFalse(replaying.exact());
    Dispatch above = path.leave(top, null, null);
    assertNotNull(above);
    assertFalse(above.exact());
    // Handed over, the replaying dispatch is kept for the rest of the walk too.
    assertNotNull(path.kept(after));
  }

  /** A dispatch that did not complete keeps nothing, and no dispatch above it is handed over. */
  @Test
  public void testAbandonedDispatch() {
    Walk walk = new Walk().source("top").source("failing");
    DispatchPath path = new DispatchPath();
    path.enter(walk.sources.get("top"));
    path.enter(walk.sources.get("failing"));
    path.abandon(walk.sources.get("failing"));
    assertNull(path.leave(walk.sources.get("top"), null, null));
    assertNull(path.kept(walk.sources.get("failing")));
  }
}
