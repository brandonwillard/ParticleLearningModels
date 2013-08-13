package utils;

import gov.sandia.cognition.collection.ArrayUtil;
import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.WeightedValue;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;

public class SamplingUtils {
  

//  public static <D> List<WeightedValue<D>> waterFillingResample(final double[] logWeights, 
//    final double logWeightSum, final List<D> domain, final Random random, final int N) {
//    Preconditions.checkArgument(domain.size() == logWeights.length);
//    Preconditions.checkArgument(logWeights.length >= N);
//
//    final List<Double> nLogWeights = Doubles.asList(logWeights);
//    final List<WeightedValue<D>> nonZeroWeights = Lists.newArrayList();
//    for (int i = 0; i < nLogWeights.size(); i++) {
//      final double logWeight = nLogWeights.get(i);
//      nLogWeights.set(i, logWeight - logWeightSum);
//      if (Double.compare(logWeight, Double.NEGATIVE_INFINITY) > 0d)
//        nonZeroWeights.add(DefaultWeightedValue.create(domain.get(i), logWeight));
//    }
//  
//    List<WeightedValue<D>> result;
////    collapsed.idx = NULL
////    collapsed.weights = NULL
////    resample.type = "water-filling"
////    non.zero.count = sum(log.q > -Inf)
////    log.alpha = NULL
//    final int nonZeroCount = nonZeroWeights.size();
//    if (nonZeroCount == N) {
////      # do nothing but remove the zero weights
//      result = nonZeroWeights;
//    } else if (nonZeroCount < N) {
////      # in this case, we need to just plain 'ol resample 
////      keepers = which(log.q > -Inf)
////      keepers.weights = log.q[keepers]
////      collapsed.weights = rep(-log(num.samples), num.samples)
////      collapsed.idx = keepers[resample(keepers.weights, 
////                                     num.samples=num.samples, log=T)$indices]
////      resample.type = "standard resample"
//    } else {
////      log.alpha = find.log.alpha(log.q, num.samples)
////      if (log.alpha == 0) {
////        # plain 'ol resample here, too 
////        collapsed.weights = rep(-log(num.samples), num.samples)
////        collapsed.idx = resample(log.q, 
////            num.samples=num.samples, log=T)$indices
////        resample.type = "(log.alpha == 0) standard resample"
////      } else {
////        log.p = pmin(log.q + log.alpha, 0)
////  
////        keepers = (log.p == 0)
////        if (!any(keepers)) {
////          # all weights are below, resample
////          collapsed.weights = rep(-log(num.samples), num.samples)
////          collapsed.idx = resample(log.q, 
////              num.samples=num.samples, log=T)$indices
////          resample.type = "(all below) standard resample"
////        } else {
////          # if we're here, then we need to resample a subset
////          # of weights that were below beta.
////          below.idx = which(!keepers)
////          below.weights = log.q[!keepers]
////          below.num = num.samples - sum(keepers)
////  
////          below.keepers.idx = NULL
////          if (below.num > 0) {
////            below.keepers.idx = below.idx[resample(below.weights, 
////                                         num.samples=below.num, log=T)$indices]
////            collapsed.weights = c(log.q[keepers], rep(-log.alpha, below.num))
////  
////            # FIXME: shouldn't have to do this...
////            weights.total = log.sum(collapsed.weights)
////            stopifnot(abs(weights.total) < 1e-7)
////            collapsed.weights = collapsed.weights - weights.total
////            resample.type = "water-filling resample"
////          } else {
////            collapsed.weights = log.q
////            resample.type = "water-filling (none below) resample"
////          }
////  
////          # indices for q of particles retained after the collapse 
////          collapsed.idx = c(which(keepers), below.keepers.idx)
////        } 
////      }
//    }
////    weights.sum = log.sum(collapsed.weights)
////    stopifnot(abs(weights.sum) < 1e-7)
////    stopifnot(length(collapsed.weights) == num.samples
////                & length(collapsed.idx) == num.samples)
////    return(data.frame(weights=collapsed.weights, 
////                      indices=collapsed.idx,
////                      log.alpha=log.alpha,
////                      resample.type=resample.type))
//  }

  /**
   * Find the log of $\alpha$: the water-filling cut-off.
   * @param logWeights
   * @param N
   * @return
   */
  public static double findLogAlpha(final double[] logWeights, final int N) {
    final int M = logWeights.length;
    final double[] sLogWeights = logWeights.clone();
    Arrays.sort(sLogWeights);
    ArrayUtil.reverse(sLogWeights);
    double logTailsum = 0d;
    double logAlpha = Math.log(N);
    int k = 0;
    int pk = k;

    while (true) {
      pk = k;
      while (k < M && logAlpha + sLogWeights[k+1] > 0) {
        k++;
        logTailsum = LogMath2.subtract(logTailsum, sLogWeights[k]);
      }
      logAlpha = Math.log(N-k) - logTailsum;
      if ( pk == k || k == M ) 
        break;
    }
  
    return logAlpha;
  }
  
  public static int sampleIndexFromLogProbabilities(final Random random, final double[] logProbs,
      double totalLogProbs) {
    double value = Math.log(random.nextDouble());
    final int lastIndex = logProbs.length - 1;
    for (int i = 0; i < lastIndex; i++) {
      value = LogMath2.subtract(value, logProbs[i] - totalLogProbs);
      if (Double.isNaN(value) || value == Double.NEGATIVE_INFINITY) {
        return i;
      }
    }
    return lastIndex;
  }

  public static <D> List<D> sampleMultipleLogScale(final double[] cumulativeLogWeights,
      final double logWeightSum, final List<D> domain, final Random random, final int numSamples) {

    int index;
    final List<D> samples = Lists.newArrayListWithCapacity(numSamples);
    for (int n = 0; n < numSamples; n++) {
      final double p = logWeightSum + Math.log(random.nextDouble());
      index = Arrays.binarySearch(cumulativeLogWeights, p);
      if (index < 0) {
        final int insertionPoint = -index - 1;
        index = insertionPoint;
      }
      samples.add(domain.get(index));
    }
    return samples;

  }

}
