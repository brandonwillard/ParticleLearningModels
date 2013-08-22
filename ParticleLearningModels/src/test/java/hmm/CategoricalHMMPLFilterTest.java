package hmm;

import static org.junit.Assert.*;
import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import org.junit.Test;

import utils.CountedDataDistribution;
import utils.LogMath2;
import utils.SamplingUtils;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.math.DoubleMath;
import com.google.common.primitives.Doubles;

public class CategoricalHMMPLFilterTest {

  /**
   * Test that the HMM filter reaches an expected level of accuracy for
   * the marginal class posterior probabilities.
   */
  @Test
  public void testMarginalClassProbabilities() {
    final Random rng = new Random();//123452388l);

    final int S = 2;
    DefaultDataDistribution<Integer> s1Likelihood = new DefaultDataDistribution<Integer>();
    s1Likelihood.increment(0, 1d/3d);
    s1Likelihood.increment(1, 1d/3d);
    s1Likelihood.increment(2, 1d/3d);
    s1Likelihood.increment(3, 1d/3d);

    HiddenMarkovModel<Integer> trueHmm = HiddenMarkovModel.createRandom(S, s1Likelihood, rng);

//    DefaultDataDistribution<Double> s2Likelihood = new DefaultDataDistribution<Double>();
//    s2Likelihood.increment(0d, 1d/3d);
//    s2Likelihood.increment(1d, 1d/3d);
//    s2Likelihood.increment(2d, 1d/3d);
//
//    DefaultDataDistribution<Double> s3Likelihood = new DefaultDataDistribution<Double>();
//    s3Likelihood.increment(0d, 1d/3d);
//    s3Likelihood.increment(1d, 1d/3d);
//    s3Likelihood.increment(2d, 1d/3d);
//    
//    List<DefaultDataDistribution<Double>> emissionFunctions = Lists.newArrayList();
//    emissionFunctions.add(s1Likelihood);
//    emissionFunctions.add(s2Likelihood);
//    emissionFunctions.add(s3Likelihood);
//    
//    HiddenMarkovModel<Double> trueHmm = new HiddenMarkovModel<Double>(
//        VectorFactory.getDefault().copyArray(new double[] {1d/4d, 1d/4d, 2d/4d}),
//        MatrixFactory.getDefault().copyArray(new double[][] {
//            // (i,j) j: from, i: to
//            {1d/3d, 1d/3d, 1d/3d}, 
//            {1d/3d, 1d/3d, 1d/3d}, 
//            {1d/3d, 1d/3d, 1d/3d}}),
//        emissionFunctions);
    
    final int T = 3;
    Pair<List<Integer>, List<Integer>> sample = SamplingUtils.sampleWithStates(trueHmm, rng, T);

    HiddenMarkovModel<Integer> hmm = trueHmm.clone();
//    HiddenMarkovModel<Double> hmm = new HiddenMarkovModel<Double>(
//        VectorFactory.getDefault().copyArray(new double[] {1d/4d, 1d/4d, 2d/4d}),
//        MatrixFactory.getDefault().copyArray(new double[][] {
//            // (i,j) j: from, i: to
//            {1d/3d, 1d/3d, 1d/3d}, 
//            {1d/3d, 1d/3d, 1d/3d}, 
//            {1d/3d, 1d/3d, 1d/3d}}),
//        emissionFunctions
//        );

    CategoricalHMMPLFilter filter = new CategoricalHMMPLFilter(hmm, rng, false);
    
    List<RingAccumulator<MutableDouble>> classErrorRates = Lists.newArrayList();
    for (int j = 0; j < hmm.getNumStates(); j++) {
      RingAccumulator<MutableDouble> cRate = new RingAccumulator<MutableDouble>();
      classErrorRates.add(cRate);
    }

    List<Vector> forwardResults = hmm.stateBeliefs(sample.getFirst());

    final int numIterations = 10;
    final int N = 10000;
    for (int k = 0; k < numIterations; k++) {
      filter.setNumParticles(N);
  
      CountedDataDistribution<HMMTransitionState<Integer>> distribution = 
          (CountedDataDistribution<HMMTransitionState<Integer>>) filter.createInitialLearnedObject();
      
      /*
       * Recurse through the particle filter
       */
      for (int i = 0; i < T; i++) {
        final Integer y = sample.getFirst().get(i);
        final ObservedState obsState = new ObservedState(i, y);
        filter.update(distribution, obsState);
  
        CountedDataDistribution<Integer> stateSums = new CountedDataDistribution<>(true);
        for (HMMTransitionState<Integer> state : distribution.getDomain()) {
          stateSums.adjust(state.getState(), distribution.getLogFraction(state), distribution.getCount(state));
        }
  
        /*
         * Check marginal state probabilities
         */
        Preconditions.checkState(stateSums.getDomainSize() <= hmm.getNumStates());
        for (int j = 0; j < hmm.getNumStates(); j++) {
          final double classError = forwardResults.get(i).getElement(j) - stateSums.getFraction(j);
          Preconditions.checkState(!Double.isNaN(classError));
          classErrorRates.get(j).accumulate(new MutableDouble(classError));
        }
      }
    }
    
    List<Double> errMeans = Lists.newArrayList();
    for (int j = 0; j < hmm.getNumStates(); j++) {
      System.out.println("meanClassProbError=" + classErrorRates.get(j).getMean().value);
      errMeans.add(classErrorRates.get(j).getMean().value);
    }
    assertArrayEquals(Doubles.toArray(Collections.nCopies(hmm.getNumStates(), 0d)), 
        Doubles.toArray(errMeans), 5e-3);
  }

  @Test
  public void testBaumWelchInitialization1() {
    DefaultDataDistribution<Double> s1Likelihood = new DefaultDataDistribution<Double>();
    s1Likelihood.increment(0d, 1d/3d);
    s1Likelihood.increment(1d, 1d/3d);

    DefaultDataDistribution<Double> s2Likelihood = new DefaultDataDistribution<Double>();
    s2Likelihood.increment(0d, 1d/3d);
    s2Likelihood.increment(1d, 1d/3d);
    
    List<DefaultDataDistribution<Double>> emissionFunctions = Lists.newArrayList();
    emissionFunctions.add(s1Likelihood);
    emissionFunctions.add(s2Likelihood);
    
    HiddenMarkovModel<Double> hmm = new HiddenMarkovModel<Double>(
        VectorFactory.getDefault().copyArray(new double[] {1d/2d, 1d/2d}),
        MatrixFactory.getDefault().copyArray(new double[][] {
            // (i,j) j: from, i: to
            {1d/3d, 1d/3d}, 
            {1d/3d, 1d/3d}}),
        emissionFunctions);
    
    TreeSet<HMMTransitionState<Double>> result = CategoricalHMMPLFilter.
        expandForwardProbabilities(hmm, Lists.newArrayList(0d, 1d));
    
    assertTrue(result.size() == 4);
    for (HMMTransitionState<Double> chain : result) {
      System.out.println(chain.getStateHistory());
      System.out.println("\t" + chain);
      assertEquals(Math.log(1d/4d), chain.getStateLogWeight(), 1e-7);
    }
    System.out.println(result);
  }
  
  @Test
  public void testBaumWelchInitialization2() {
    DefaultDataDistribution<Double> s1Likelihood = new DefaultDataDistribution<Double>();
    s1Likelihood.increment(0d, 1d/2d);
    s1Likelihood.increment(1d, 1d/2d);

    DefaultDataDistribution<Double> s2Likelihood = new DefaultDataDistribution<Double>();
    s2Likelihood.increment(0d, 2d/3d);
    s2Likelihood.increment(1d, 1d/3d);
    
    List<DefaultDataDistribution<Double>> emissionFunctions = Lists.newArrayList();
    emissionFunctions.add(s1Likelihood);
    emissionFunctions.add(s2Likelihood);
    
    HiddenMarkovModel<Double> hmm = new HiddenMarkovModel<Double>(
        VectorFactory.getDefault().copyArray(new double[] {1d/2d, 1d/2d}),
        MatrixFactory.getDefault().copyArray(new double[][] {
            // (i,j) j: from, i: to
            {9d/10d, 1d/10d}, 
            {1d/10d, 9d/10d}}),
        emissionFunctions);
    
    ArrayList<Double> observations = Lists.newArrayList(0d, 1d);
    TreeSet<HMMTransitionState<Double>> result = CategoricalHMMPLFilter.
        expandForwardProbabilities(hmm, observations);
    
    assertTrue(result.size() == Math.pow(hmm.getNumStates(), observations.size()));
    double totalLogLikelihood = Double.NEGATIVE_INFINITY;
    Set<Double> uniqueValues = Sets.newHashSet();
    for (HMMTransitionState<Double> chain : result) {
      System.out.println(chain.getStateHistory());
      System.out.println("\t" + chain);
      uniqueValues.add(chain.getStateLogWeight());
      totalLogLikelihood = LogMath2.add(totalLogLikelihood, chain.getStateLogWeight());
//      assertEquals(Math.log(1d/4d), chain.getStateLogWeight(), 1e-7);
    }
    assertEquals(0d, totalLogLikelihood, 1e-7);
    System.out.println("uniquely weighed chains = " + uniqueValues.size()
        + "/" + result.size());
  }
  
  /**
   * Test that the HMM filter reaches an expected level of accuracy for
   * the marginal class posterior probabilities.  Initialized with
   * Baum-Welch-derived particles.
   */
  @Test
  public void testMarginalClassProbabilities2() {
    final Random rng = new Random();//123452388l);

    final int S = 2;
    DefaultDataDistribution<Integer> s1Likelihood = new DefaultDataDistribution<Integer>();
    s1Likelihood.increment(0, 1d/2d);
    s1Likelihood.increment(1, 1d/2d);

    HiddenMarkovModel<Integer> trueHmm = HiddenMarkovModel.createRandom(S, s1Likelihood, rng);
    
    final int T = 1;
    Pair<List<Integer>, List<Integer>> sample = SamplingUtils.sampleWithStates(trueHmm, rng, T);

    HiddenMarkovModel<Integer> hmm = trueHmm.clone();

    CategoricalHMMPLFilter filter = new CategoricalHMMPLFilter(hmm, rng, false);
    
    List<RingAccumulator<MutableDouble>> classErrorRates = Lists.newArrayList();
    for (int j = 0; j < hmm.getNumStates(); j++) {
      RingAccumulator<MutableDouble> cRate = new RingAccumulator<MutableDouble>();
      classErrorRates.add(cRate);
    }

    List<Vector> forwardResults = hmm.stateBeliefs(sample.getFirst());

    final int numIterations = 2;
    final int N = 10000;
    for (int k = 0; k < numIterations; k++) {
      filter.setNumParticles(N);
  
      CountedDataDistribution<HMMTransitionState<Integer>> distribution = 
          (CountedDataDistribution<HMMTransitionState<Integer>>) filter.createInitialLearnedObject();
      
      /*
       * Recurse through the particle filter
       */
      for (int i = 0; i < T; i++) {
        final Integer y = sample.getFirst().get(i);
        final ObservedState obsState = new ObservedState(i, y);
        filter.update(distribution, obsState);
  
        CountedDataDistribution<Integer> stateSums = new CountedDataDistribution<>(true);
        for (HMMTransitionState<Integer> state : distribution.getDomain()) {
          stateSums.adjust(state.getState(), distribution.getLogFraction(state), distribution.getCount(state));
        }
  
        /*
         * Check marginal state probabilities
         */
        Preconditions.checkState(stateSums.getDomainSize() <= hmm.getNumStates());
        for (int j = 0; j < hmm.getNumStates(); j++) {
          final double classError = forwardResults.get(i).getElement(j) - stateSums.getFraction(j);
          Preconditions.checkState(!Double.isNaN(classError));
          classErrorRates.get(j).accumulate(new MutableDouble(classError));
        }
      }
    }
    
    List<Double> errMeans = Lists.newArrayList();
    for (int j = 0; j < hmm.getNumStates(); j++) {
      System.out.println("meanClassProbError=" + classErrorRates.get(j).getMean().value);
      errMeans.add(classErrorRates.get(j).getMean().value);
    }
    assertArrayEquals(Doubles.toArray(Collections.nCopies(hmm.getNumStates(), 0d)), 
        Doubles.toArray(errMeans), 5e-3);
  }
  
}
