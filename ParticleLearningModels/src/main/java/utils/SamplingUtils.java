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
import hmm.HmmTransitionState;

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

    final List<Double> nonZeroLogWeights = Lists.newArrayList();
    final List<Double> cumNonZeroLogWeights = Lists.newArrayList();
    final List<D> nonZeroObjects = Lists.newArrayList();
    double nonZeroTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < logWeights.length; i++) {
      final double normedLogWeight = logWeights[i] - logWeightSum;
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
      log.debug("removed zero weights");

//    } else if (nonZeroCount < N) {
//      /*
//       * In this case, we need to just plain 'ol resample 
//       */
//      resultObjects = sampleMultipleLogScale(Doubles.toArray(cumNonZeroWeights), 
//          nonZeroTotal, nonZeroObjects, random, N, false);
//      resultWeights = Collections.nCopies(N, -Math.log(N));
//      log.warn("non-zero less than N");
    } else {

      final double logAlpha = findLogAlpha(Doubles.toArray(nonZeroLogWeights),
          nonZeroTotal, N);
      if (logAlpha == 0 || Double.isNaN(logAlpha)) {
        /*
         * Plain 'ol resample here, too 
         */
        resultObjects = sampleNoReplaceMultipleLogScale(Doubles.toArray(nonZeroLogWeights), 
            nonZeroTotal, nonZeroObjects, random, N);
        resultLogWeights = Collections.nCopies(N, -Math.log(N));
        log.warn("logAlpha = 0");
      } else {

        List<Double> logPValues = Lists.newArrayListWithCapacity(nonZeroCount);
        List<Double> keeperLogWeights = Lists.newArrayList();
        List<D> keeperObjects = Lists.newArrayList();
        List<Double> belowLogWeights = Lists.newArrayList();
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
            belowLogWeights.add(logQ);
          }
        }

        if (keeperLogWeights.isEmpty()) {
          /*
           * All weights are below, resample
           */
          resultObjects = sampleNoReplaceMultipleLogScale(Doubles.toArray(nonZeroLogWeights), 
              nonZeroTotal, nonZeroObjects, random, N);
          resultLogWeights = Collections.nCopies(N, -Math.log(N));
          log.debug("all below logAlpha");
        } else {
          wasWaterFillingApplied = true;
          log.debug("water-filling applied!");
          if (!belowLogWeights.isEmpty()) {
            /*
             * Resample the below beta entries
             */
            final int resampleN = N - keeperLogWeights.size();
            List<D> belowObjectsResampled = sampleNoReplaceMultipleLogScale(Doubles.toArray(belowLogWeights), 
                belowPTotal, belowObjects, random, resampleN);
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
   * Find the log of alpha: the water-filling cut-off.
   * The logTotalWeight is important for numerical stability.  Since
   * we're looping through and subtracting log values, we need
   * to know what "zero" is (relative to the numeric precision). 
   * 
   * @param logWeights
   * @param logTotalWeight
   * @param N
   * @return
   */
  public static double findLogAlpha(final double[] logWeights, final double logTotalWeight, final int N) {
    final int M = logWeights.length;
    final double[] sLogWeights = logWeights.clone();
    Arrays.sort(sLogWeights);
    ArrayUtil.reverse(sLogWeights);
    double logTailSum = Math.abs(logTotalWeight);
    double logAlpha = Math.log(N);
    int k = 0;
    int pk = k;
    
    Preconditions.checkArgument(SamplingUtils.isLogNormalized(sLogWeights, 1e-7));

    while (true) {
      pk = k;
      while (k < M && logAlpha + sLogWeights[k] > 0) {
        final double thisLogWeight = sLogWeights[k];
        final double logTailSumTmp;
        /*
         * TODO FIXME XXX: terrible hack!  fix this!
         */
        if (logTailSum < thisLogWeight) {
          final double tmpDiff = Math.exp(logTailSum) - Math.exp(thisLogWeight);
          if (Math.abs(tmpDiff) < 1e-1) {
            logTailSumTmp = Double.NEGATIVE_INFINITY;
            log.warn("numerical instability");
          } else {
            logTailSumTmp = LogMath2.subtract(logTailSum, thisLogWeight);
          }
        } else {
          logTailSumTmp = LogMath2.subtract(logTailSum, thisLogWeight);
        }

        Preconditions.checkState(!Double.isNaN(logTailSumTmp));
        logTailSum = logTailSumTmp;
        k++;
      }
      logAlpha = Math.log(N-k) - logTailSum;
      if ( pk == k || k == M ) 
        break;
    }
    
//    /*
//     * Here's a special case when we have N-many disproportionately 
//     * large and equal weights...we just reset alpha so that resampling
//     * is required.  This wouldn't be good if we wanted to know something
//     * more about our weights, but we don't need to, yet. 
//     */
//    if (k == N && Double.isNaN(logAlpha))
//      return logAlpha;
//    else
//      Preconditions.checkState(!Double.isNaN(logAlpha));
  
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
  
  /**
   * Sort a[] into descending order by "heapsort";
   * sort ib[] alongside;
   * if initially, ib[] = 1...n, it will contain the permutation finally
   *
   * From R's <a href="https://github.com/wch/r-source/blob/2633f0ee6306b23eadda989110f5748ce4c050bd/src/main/sort.c#L264">source code</a>.
   * @param a
   * @param ib
   * @param n
   */
  public static void revsort(double[] a, int[] ib, int n) {
    /*
     * Sort a[] into descending order by "heapsort"; sort ib[] alongside; if
     * initially, ib[] = 1...n, it will contain the permutation finally
     */

    int l, j, ir, i;
    double ra;
    int ii;

    if (n <= 1)
      return;

//    a--;
//    ib--;

    l = (n >> 1) + 1;
    ir = n;

    for (;;) {
      if (l > 1) {
        l = l - 1;
        ra = a[l - 1];
        ii = ib[l - 1];
      } else {
        ra = a[ir - 1];
        ii = ib[ir - 1];
        a[ir - 1] = a[0];
        ib[ir - 1] = ib[0];
        if (--ir == 1) {
          a[0] = ra;
          ib[0] = ii;
          return;
        }
      }
      i = l;
      j = l << 1;
      while (j <= ir) {
        if (j < ir && a[j - 1] > a[j])
          ++j;
        if (ra > a[j - 1]) {
          a[i - 1] = a[j - 1];
          ib[i - 1] = ib[j - 1];
          j += (i = j);
        } else
          j = ir + 1;
      }
      a[i - 1] = ra;
      ib[i - 1] = ii;
    }
  }

  /**
   * Weighed sampling without replacement from R's 
   * <a href="https://github.com/wch/r-source/blob/trunk/src/main/random.c">source</a>.
   * 
   * @param n
   * @param p
   * @param perm
   * @param nans
   * @param ans
   * @param rng
   */
  static void ProbSampleNoReplace(int n, double[] p, int[] perm, int nans,
      int[] ans, Random rng) {
    double rT, mass, totalmass;
    int i, j, k, n1;

    /* Record element identities */
    for (i = 0; i < n; i++)
      perm[i] = i + 1;

    /* Sort probabilities into descending order */
    /* Order element identities in parallel */
    revsort(p, perm, n);

    /* Compute the sample */
    totalmass = 1;
    for (i = 0, n1 = n - 1; i < nans; i++, n1--) {
      rT = totalmass * rng.nextDouble();
      mass = 0;
      for (j = 0; j < n1; j++) {
        mass += p[j];
        if (rT <= mass)
          break;
      }
      ans[i] = perm[j];
      totalmass -= p[j];
      for (k = j; k < n1; k++) {
        p[k] = p[k + 1];
        perm[k] = perm[k + 1];
      }
    }
  }

  /**
   * Resample without replacement based on 
   * <a href="https://www.sciencedirect.com/science/article/pii/S002001900500298X">
   *  Pavlos S. Efraimidis, Paul G. Spirakis, Weighted random sampling with a reservoir</a>
   *  from the discussion <a href="http://stackoverflow.com/questions/15113650/faster-weighted-sampling-without-replacement">
   *  here</a>
   * @param logWeights
   * @param logWeightSum
   * @param domain
   * @param random
   * @param numSamples
   * @return
   */
  public static <D> List<D> sampleNoReplaceMultipleLogScale(final double[] logWeights,
      final double logWeightSum, final List<D> domain, final Random random, final int numSamples) {

    Preconditions.checkArgument(domain.size() >= numSamples);
    
    if (domain.size() == numSamples) {
      return domain;
    }
    
    double[] weights = new double[logWeights.length];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = Math.exp(logWeights[i] - logWeightSum);
    }
    int[] perm = new int[logWeights.length];
    int[] ans = new int[numSamples];
    ProbSampleNoReplace(logWeights.length, weights, perm, numSamples, ans, random);
    final List<D> samples = Lists.newArrayListWithCapacity(numSamples);
    for (int i = 0; i < ans.length; i++) {
      samples.add(domain.get(ans[i]-1));
    }
    
//    TreeMap<Double, D> tMap = Maps.newTreeMap(new Comparator<Double>() {
//      @Override
//      public int compare(Double o1, Double o2) {
//        return (o1 > o2) ? -1 : 1;
//      }
//      
//    });
//    double[] key = new double[logWeights.length];
//    for (int i = 0; i < logWeights.length; i++) {
//      final double logWeight = logWeights[i] - logWeightSum;
//      key[i] = Math.log(random.nextDouble())/logWeight;
//      tMap.put(key[i], domain.get(i));
//    }
//    
//    while(samples.size() < numSamples) {
//      samples.add(tMap.pollFirstEntry().getValue());
//    }
    
    return samples;
  }

  public static <D> List<D> sampleMultipleLogScale(final double[] cumulativeLogWeights,
      final double logWeightSum, final List<D> domain, final Random random, final int numSamples) {
    Preconditions.checkArgument(domain.size() == cumulativeLogWeights.length);

    final List<D> samples = Lists.newArrayListWithCapacity(numSamples);
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
