package utils;

import gov.sandia.cognition.collection.ArrayUtil;
import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.util.DefaultPair;
import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;

public class SamplingUtils {
  

  public static <D> Pair<List<Double>, List<D>> waterFillingResample(final double[] logWeights, 
    final double logWeightSum, final List<D> domain, final Random random, final int N) {
    Preconditions.checkArgument(domain.size() == logWeights.length);
    Preconditions.checkArgument(logWeights.length >= N);

    final List<Double> nLogWeights = Doubles.asList(logWeights);
    final List<Double> nonZeroWeights = Lists.newArrayList();
    final List<D> nonZeroObjects = Lists.newArrayList();
    double nonZeroTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < nLogWeights.size(); i++) {
      final double logWeight = nLogWeights.get(i);
      nLogWeights.set(i, logWeight - logWeightSum);
      if (Double.compare(logWeight, Double.NEGATIVE_INFINITY) > 0d) {
        nonZeroObjects.add(domain.get(i));
        nonZeroWeights.add(logWeight);
        nonZeroTotal = LogMath2.add(nonZeroTotal, logWeight);
      }
    }
  
    List<Double> resultWeights;
    List<D> resultObjects;
    final int nonZeroCount = nonZeroWeights.size();
    if (nonZeroCount == N) {
      /*
       * Do nothing but remove the zero weights
       */
      resultWeights = nonZeroWeights;
      resultObjects = nonZeroObjects;
    } else if (nonZeroCount < N) {
      /*
       * In this case, we need to just plain 'ol resample 
       */
      resultObjects = sampleMultipleLogScale(Doubles.toArray(nonZeroWeights), 
          nonZeroTotal, nonZeroObjects, random, N);
      resultWeights = Lists.newArrayListWithCapacity(nonZeroCount);
      Collections.fill(resultWeights, -Math.log(nonZeroCount));
    } else {
      final double logAlpha = findLogAlpha(Doubles.toArray(nonZeroWeights), N);
      if (logAlpha == 0) {
        /*
         * Plain 'ol resample here, too 
         */
        resultObjects = sampleMultipleLogScale(Doubles.toArray(nonZeroWeights), 
            nonZeroTotal, nonZeroObjects, random, N);
        resultWeights = Lists.newArrayListWithCapacity(N);
        Collections.fill(resultWeights, -Math.log(N));
      } else {

        List<Double> logPValues = Lists.newArrayListWithCapacity(nonZeroCount);
        List<Double> keeperLogWeights = Lists.newArrayList();
        List<D> keeperObjects = Lists.newArrayList();
        List<Double> belowLogWeights = Lists.newArrayList();
        List<D> belowObjects = Lists.newArrayList();
        double belowPTotal = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < nonZeroWeights.size(); j++) {
          final double logQ = nonZeroWeights.get(j);
          final double logP = Math.min(logQ + logAlpha, 0d);
          final D object = nonZeroObjects.get(j);
          logPValues.add(logP);
          if (logP == 0d) {
            keeperLogWeights.add(logQ);
            keeperObjects.add(object);
          } else {
            belowLogWeights.add(logQ);
            belowObjects.add(object);
            belowPTotal = LogMath2.add(belowPTotal, logQ);
          }
        }

        if (keeperLogWeights.isEmpty()) {
          /*
           * All weights are below, resample
           */
          resultObjects = sampleMultipleLogScale(Doubles.toArray(nonZeroWeights), 
              nonZeroTotal, nonZeroObjects, random, N);
          resultWeights = Lists.newArrayListWithCapacity(N);
          Collections.fill(resultWeights, -Math.log(N));
        } else {
          if (!belowLogWeights.isEmpty()) {
            /*
             * Resample the below beta entries
             */
            Preconditions.checkState(N - belowLogWeights.size() > 0);
            List<D> belowObjectsResampled = sampleMultipleLogScale(Doubles.toArray(belowLogWeights), 
                belowPTotal, belowObjects, random, N - belowLogWeights.size());
            List<Double> belowWeightsResampled = Lists.newArrayListWithCapacity(N);
            Collections.fill(belowWeightsResampled, -Math.log(nonZeroCount));
            
            keeperObjects.addAll(belowObjectsResampled);
            keeperLogWeights.addAll(belowWeightsResampled);
          } 
          
          Preconditions.checkState(isLogNormalized(keeperLogWeights, 1e-7));

          resultObjects = keeperObjects;
          resultWeights = keeperLogWeights;
        } 
      }
    }
    
    Preconditions.checkState(resultWeights.size() == resultObjects.size()
        && resultWeights.size() == N);
    return DefaultPair.create(resultWeights, resultObjects);
  }

  /**
   * Checks that weights are normalized up to the given magnitude.
   * 
   * @param logWeights
   * @param zeroPrec
   * @return
   */
  public static boolean
      isLogNormalized(final List<Double> logWeights, final double zeroPrec) {
    return isLogNormalized(Doubles.toArray(logWeights), zeroPrec);
  }

  /**
   * Checks that weights are normalized up to the given magnitude.
   * 
   * @param logWeights
   * @param zeroPrec
   * @return
   */
  public static boolean
      isLogNormalized(final double[] logWeights, final double zeroPrec) {
    double logTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < logWeights.length; i++) {
      logTotal = LogMath2.add(logTotal, logWeights[i]);
    }
    return Math.abs(logTotal) < zeroPrec;
  }

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
