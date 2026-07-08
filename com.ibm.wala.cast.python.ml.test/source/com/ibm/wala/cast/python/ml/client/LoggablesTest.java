package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.util.intset.OrdinalSet;
import org.junit.Test;

/**
 * Unit tests for {@link Loggables}, guarding the null-safety of its context-free renderers. {@code
 * findCreator} calls these with a {@code null} source, so a regression to the pre-guard form would
 * throw a {@link NullPointerException} at FINE-level logging; see <a
 * href="https://github.com/wala/ML/issues/697">wala/ML#697</a>.
 */
public class LoggablesTest {

  @Test
  public void describeNullPointsToSetVariable() {
    assertEquals("null", Loggables.describe((PointsToSetVariable) null));
  }

  @Test
  public void describeNullPointerKey() {
    assertEquals("null", Loggables.describe((PointerKey) null));
  }

  @Test
  public void describeNullCGNode() {
    assertEquals("null", Loggables.describe((CGNode) null));
  }

  @Test
  public void describeNullInstanceKey() {
    assertEquals("null", Loggables.describe((InstanceKey) null));
  }

  @Test
  public void describeNullOrdinalSet() {
    assertEquals("null", Loggables.describe((OrdinalSet<InstanceKey>) null));
  }

  @Test
  public void describeEmptyOrdinalSet() {
    assertEquals("[]", Loggables.describe(OrdinalSet.empty()));
  }
}
