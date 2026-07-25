package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * A generator for the shape-and-dtype-preserving {@code tf.image} augmentation ops ({@code
 * random_flip_left_right} and the {@code adjust_contrast}/{@code adjust_brightness}/{@code
 * adjust_saturation}/{@code adjust_hue} family): each returns a fresh tensor of the same shape and
 * dtype as its image argument, with every other argument a scalar. One class serves all five op
 * types, the {@link ElementWiseOperation} precedent for a shared generator over several dispatch
 * keys. See wala/ML#792.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ImageAugmentation extends PassThroughUnaryTensorGenerator {

  /**
   * Constructs a new {@code ImageAugmentation}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  public ImageAugmentation(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a new {@code ImageAugmentation}.
   *
   * @param node The synthetic {@link CGNode} allocating the result.
   */
  public ImageAugmentation(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "image";
  }
}
