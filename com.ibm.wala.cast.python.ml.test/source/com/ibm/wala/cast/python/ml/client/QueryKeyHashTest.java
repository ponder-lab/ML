package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.util.collections.Pair;
import java.lang.reflect.Proxy;
import org.junit.Test;

/**
 * Tests that the resolver's query keys compute their hash codes once and keep the values and the
 * equality they had (wala/ML#1176): a {@link Pair}'s for {@link HashedPair}, and a record's of the
 * same components for {@link TensorGenerator.ShapeQuery} and {@link TensorGenerator.BoundKey}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class QueryKeyHashTest {

  /** A component that counts how often its hash code is computed. */
  private static final class Counted {

    private int hashes;

    @Override
    public int hashCode() {
      this.hashes++;
      return 1_000_003;
    }
  }

  /** The record a shape query was, for its hash code. */
  private record ShapeQueryRecord(CGNode node, int valueNumber, boolean exact) {}

  /** The record a bound key was, for its hash code. */
  private record BoundKeyRecord(TensorGenerator.ValueBinding binding, Object key) {}

  /**
   * Makes a call-graph node whose hash code is fixed and counted.
   *
   * @param hashes The count of hash code computations.
   * @return The node.
   */
  private static CGNode countedNode(int[] hashes) {
    return (CGNode)
        Proxy.newProxyInstance(
            CGNode.class.getClassLoader(),
            new Class<?>[] {CGNode.class},
            (proxy, method, args) ->
                switch (method.getName()) {
                  case "hashCode" -> {
                    hashes[0]++;
                    yield 77_777;
                  }
                  case "equals" -> proxy == args[0];
                  case "toString" -> "node";
                  default -> throw new UnsupportedOperationException(method.getName());
                });
  }

  /** A hashed pair hashes and compares as a plain pair of the same components does. */
  @Test
  public void testHashedPairMatchesPair() {
    Object nested = Pair.make("inner", 3);
    HashedPair<String, Object> hashed = HashedPair.of("shape-eval", nested);
    Pair<String, Object> plain = Pair.make("shape-eval", nested);

    assertEquals(plain.hashCode(), hashed.hashCode());
    assertEquals(plain, hashed);
    assertEquals(hashed, plain);
    assertNotEquals(Pair.make("dtype-eval", nested), hashed);
  }

  /** A hashed pair computes its components' hash codes once, however often it is hashed. */
  @Test
  public void testHashedPairHashesOnce() {
    Counted component = new Counted();
    HashedPair<Counted, String> hashed = HashedPair.of(component, "tag");

    hashed.hashCode();
    hashed.hashCode();

    assertEquals(1, component.hashes);
  }

  /** A shape query hashes as the record of its components did, and computes that once. */
  @Test
  public void testShapeQueryMatchesRecordAndHashesOnce() {
    int[] hashes = {0};
    CGNode node = countedNode(hashes);
    TensorGenerator.ShapeQuery query = new TensorGenerator.ShapeQuery(node, 5, true);

    assertEquals(1, hashes[0]);
    query.hashCode();
    query.hashCode();
    assertEquals(1, hashes[0]);

    assertEquals(new ShapeQueryRecord(node, 5, true).hashCode(), query.hashCode());
    assertEquals(
        new ShapeQueryRecord(null, -2, false).hashCode(),
        new TensorGenerator.ShapeQuery(null, -2, false).hashCode());
    assertEquals(new TensorGenerator.ShapeQuery(node, 5, true), query);
    assertNotEquals(new TensorGenerator.ShapeQuery(node, 5, false), query);
    assertNotEquals(new TensorGenerator.ShapeQuery(node, 6, true), query);
  }

  /** A bound key hashes as the record of its components did, and computes that once. */
  @Test
  public void testBoundKeyMatchesRecordAndHashesOnce() {
    Counted inner = new Counted();
    TensorGenerator.BoundKey bound = new TensorGenerator.BoundKey(null, inner);

    assertEquals(1, inner.hashes);
    bound.hashCode();
    bound.hashCode();
    assertEquals(1, inner.hashes);

    assertEquals(new BoundKeyRecord(null, inner).hashCode(), bound.hashCode());
    assertEquals(new TensorGenerator.BoundKey(null, inner), bound);
    assertNotEquals(new TensorGenerator.BoundKey(null, "other"), bound);
  }
}
