package utils;

import gov.sandia.cognition.collection.ArrayUtil;
import gov.sandia.cognition.collection.CollectionUtil;
import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
import gov.sandia.cognition.statistics.bayesian.BayesianUtil;
import gov.sandia.cognition.statistics.distribution.CategoricalDistribution;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.ExponentialDistribution;
import gov.sandia.cognition.util.DefaultPair;
import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;
import hmm.HMMTransitionState;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.TreeMap;

import org.apache.log4j.Logger;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.primitives.Doubles;

public class SamplingUtils {
  
  final static Logger log = Logger
      .getLogger(SamplingUtils.class);

  public static <ObservationType> Pair<List<ObservationType>, List<Integer>> sampleWithStates(
      HiddenMarkovModel<ObservationType> hmm, Random random, int numSamples) {
  
    List<ObservationType> samples =
        Lists.newArrayListWithCapacity(numSamples);
    List<Integer> states =
        Lists.newArrayListWithCapacity(numSamples);
    Vector p = hmm.getInitialProbability();
    int state = -1;
    for (int n = 0; n < numSamples; n++) {
      double value = random.nextDouble();
      state = -1;
      while (value > 0.0) {
        state++;
        value -= p.getElement(state);
      }
  
      ObservationType sample =
          CollectionUtil.getElement(hmm.getEmissionFunctions(),
              state).sample(random);
      states.add(new Integer(state));
      samples.add(sample);
      p = hmm.getTransitionProbability().getColumn(state);
    }
  
    Pair<List<ObservationType>, List<Integer>> result = DefaultPair.create(samples, states);
    return result;
  }

  public static <D> WFCountedDataDistribution<D> waterFillingResample(final double[] logWeights, 
    final double logWeightSum, final List<D> domain, final Random random, final int N) {
    Preconditions.checkArgument(domain.size() == logWeights.length);
    Preconditions.checkArgument(logWeights.length >= N);

    final List<Double> nLogWeights = Doubles.asList(logWeights);
    final List<Double> nonZeroLogWeights = Lists.newArrayList();
    final List<Double> cumNonZeroLogWeights = Lists.newArrayList();
    final List<D> nonZeroObjects = Lists.newArrayList();
    double nonZeroTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < nLogWeights.size(); i++) {
      final double normedLogWeight = nLogWeights.get(i) - logWeightSum;
      nLogWeights.set(i, normedLogWeight);
      if (Double.compare(normedLogWeight, Double.NEGATIVE_INFINITY) > 0d) {
        nonZeroObjects.add(domain.get(i));
        nonZeroLogWeights.add(normedLogWeight);
        nonZeroTotal = LogMath2.add(nonZeroTotal, normedLogWeight);
        cumNonZeroLogWeights.add(nonZeroTotal);
      }
    }
    
    Preconditions.checkState(Math.abs(Iterables.getLast(cumNonZeroLogWeights)) < 1e-7);
    
    boolean wasWaterFillingApplied = false;
    List<Double> resultLogWeights;
    List<D> resultObjects;
    final int nonZeroCount = nonZeroLogWeights.size();
    Preconditions.checkState(nonZeroCount >= N);
    if (nonZeroCount == N) {
      /*
       * Do nothing but remove the zero weights
       */
      resultLogWeights = nonZeroLogWeights;
      resultObjects = nonZeroObjects;
      log.warn("removed zero weights");

//    } else if (nonZeroCount < N) {
//      /*
//       * In this case, we need to just plain 'ol resample 
//       */
//      resultObjects = sampleMultipleLogScale(Doubles.toArray(cumNonZeroWeights), 
//          nonZeroTotal, nonZeroObjects, random, N, false);
//      resultWeights = Collections.nCopies(N, -Math.log(N));
//      log.warn("non-zero less than N");
    } else {

      final double logAlpha = findLogAlpha(Doubles.toArray(nonZeroLogWeights), N);
      if (logAlpha == 0) {
        /*
         * Plain 'ol resample here, too 
         */
        resultObjects = sampleMultipleLogScale(Doubles.toArray(cumNonZeroLogWeights), 
            nonZeroTotal, nonZeroObjects, random, N, false);
        resultLogWeights = Collections.nCopies(N, -Math.log(N));
        log.warn("logAlpha = 0");
      } else {

        List<Double> logPValues = Lists.newArrayListWithCapacity(nonZeroCount);
        List<Double> keeperLogWeights = Lists.newArrayList();
        List<D> keeperObjects = Lists.newArrayList();
        List<Double> cummBelowLogWeights = Lists.newArrayList();
        List<D> belowObjects = Lists.newArrayList();
        double belowPTotal = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < nonZeroLogWeights.size(); j++) {
          final double logQ = nonZeroLogWeights.get(j);
          final double logP = Math.min(logQ + logAlpha, 0d);
          final D object = nonZeroObjects.get(j);
          logPValues.add(logP);
          if (logP == 0d) {
            keeperLogWeights.add(logQ);
            keeperObjects.add(object);
          } else {
            belowObjects.add(object);
            belowPTotal = LogMath2.add(belowPTotal, logQ);
            cummBelowLogWeights.add(belowPTotal);
          }
        }

        if (keeperLogWeights.isEmpty()) {
          /*
           * All weights are below, resample
           */
          resultObjects = sampleMultipleLogScale(Doubles.toArray(cumNonZeroLogWeights), 
              nonZeroTotal, nonZeroObjects, random, N, false);
          resultLogWeights = Collections.nCopies(N, -Math.log(N));
          log.warn("all below logAlpha");
        } else {
          wasWaterFillingApplied = true;
          log.debug("water-filling applied!");
          if (!cummBelowLogWeights.isEmpty()) {
            /*
             * Resample the below beta entries
             */
            final int resampleN = N - keeperLogWeights.size();
            List<D> belowObjectsResampled = sampleMultipleLogScale(Doubles.toArray(cummBelowLogWeights), 
                belowPTotal, belowObjects, random, resampleN, false);
            List<Double> belowWeightsResampled = Collections.nCopies(resampleN, -logAlpha);
            
            keeperObjects.addAll(belowObjectsResampled);
            keeperLogWeights.addAll(belowWeightsResampled);
          } 
          
          Preconditions.checkState(isLogNormalized(keeperLogWeights, 1e-7));

          resultObjects = keeperObjects;
          resultLogWeights = keeperLogWeights;
        } 
      }
    }
    
    Preconditions.checkState(resultLogWeights.size() == resultObjects.size()
        && resultLogWeights.size() == N);
    WFCountedDataDistribution<D> result = new WFCountedDataDistribution<D>(N, true);
    result.setWasWaterFillingApplied(wasWaterFillingApplied);
    for (int i = 0; i < N; i++) {
      result.increment(resultObjects.get(i), resultLogWeights.get(i));
    }
    return result;
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
    Preconditions.checkArgument(zeroPrec > 0d && zeroPrec < 1e-3);
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
    
    Preconditions.checkArgument(SamplingUtils.isLogNormalized(sLogWeights, 1e-7));

    while (true) {
      pk = k;
      while (k < M && logAlpha + sLogWeights[k] > 0) {
        final double thisLogWeight = sLogWeights[k];
        logTailsum = LogMath2.subtract(logTailsum, thisLogWeight);
        k++;
      }
      logAlpha = Math.log(N-k) - logTailsum;
      if ( pk == k || k == M ) 
        break;
    }
    
    Preconditions.checkState(!Double.isNaN(logAlpha));
  
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
      final double logWeightSum, final List<D> domain, final Random random, final int numSamples,
      final boolean replace) {
    Preconditions.checkArgument(domain.size() == cumulativeLogWeights.length);

    final List<D> samples = Lists.newArrayListWithCapacity(numSamples);
    if (!replace) {
      Preconditions.checkArgument(domain.size() >= numSamples);
      
      if (domain.size() == numSamples) {
        return domain;
      }
      
      TreeMap<Double, D> tMap = Maps.newTreeMap();
      double[] key = new double[cumulativeLogWeights.length];
      key[0] = Math.log(random.nextDouble())/(cumulativeLogWeights[0] - logWeightSum);
      for (int i = 1; i < cumulativeLogWeights.length; i++) {
        final double logWeight = LogMath2.subtract(cumulativeLogWeights[i],
            cumulativeLogWeights[i-1]) - logWeightSum;
        key[i] = Math.log(random.nextDouble())/Math.exp(logWeight);
        tMap.put(key[i], domain.get(i));
      }
      
      while(samples.size() < numSamples) {
        samples.add(tMap.pollLastEntry().getValue());
      }
      
    } else {
      int index;
      for (int n = 0; n < numSamples; n++) {
        final double p = logWeightSum + Math.log(random.nextDouble());
        index = Arrays.binarySearch(cumulativeLogWeights, p);
        if (index < 0) {
          final int insertionPoint = -index - 1;
          index = insertionPoint;
        }
        samples.add(domain.get(index));
      }
    }
    return samples;

  }

  public static double logSum(final double[] logWeights) {
    double pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < logWeights.length; i++) {
      pTotal =
          LogMath2.add(pTotal, logWeights[i]);
    }
    return pTotal;
  }

  public static void logNormalize(final double[] logWeights) {
    final double totalLogWeights = SamplingUtils.logSum(
        logWeights);
    for (int i = 0; i < logWeights.length; i++) {
      logWeights[i] -= totalLogWeights;
    }
  }

  public static double[] accumulate(List<Double> logLikelihoods) {
    double pTotal = Double.NEGATIVE_INFINITY;
    double[] result = new double[logLikelihoods.size()];
    for (int i = 0; i < logLikelihoods.size(); i++) {
      pTotal =
          LogMath2.add(pTotal, logLikelihoods.get(i));
      result[i] = pTotal;
    }
    return result;
  }

}
