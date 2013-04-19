package utils;

import gov.sandia.cognition.math.LogMath;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import clustering.MvGaussianDPDistribution;

import com.google.common.collect.Lists;

public class SamplingUtils {

  public static int sampleIndexFromLogProbabilities(final Random random, final double[] logProbs, double totalLogProbs) {
    double value = Math.log(random.nextDouble());
    final int lastIndex = logProbs.length - 1;
    for (int i = 0; i < lastIndex; i++) {
      value = LogMath.subtract(value, logProbs[i] - totalLogProbs);
      if (Double.isNaN(value) || value == Double.NEGATIVE_INFINITY) {
        return i;
      }
    }
    return lastIndex;
  }

  public static List<MvGaussianDPDistribution> sampleMultipleLogScale(
    final double[] cumulativeLogWeights, final double logWeightSum,
    final List<MvGaussianDPDistribution> domain, final Random random,
    final int numSamples) {
  
    int index;
    List<MvGaussianDPDistribution> samples = Lists.newArrayListWithCapacity(numSamples);
    for (int n = 0; n < numSamples; n++) {
      double p = logWeightSum + Math.log(random.nextDouble());
      index = Arrays.binarySearch(cumulativeLogWeights, p);
      if (index < 0) {
        int insertionPoint = -index - 1;
        index = insertionPoint;
      }
      samples.add(domain.get(index));
    }
    return samples;
  
  }

}
