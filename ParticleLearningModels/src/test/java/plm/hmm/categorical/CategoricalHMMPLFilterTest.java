package plm.hmm.categorical;

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

import plm.hmm.GenericHMM.SimHmmObservedValue;
import plm.hmm.HmmTransitionState;
import plm.hmm.StandardHMM;
import plm.hmm.categorical.CategoricalHmmPlFilter;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.math.DoubleMath;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.ExtLogMath;
import com.statslibextensions.statistics.CountedDataDistribution;
import com.statslibextensions.statistics.ExtSamplingUtils;
import com.statslibextensions.util.ObservedValue;

public class CategoricalHMMPLFilterTest {

  /**
   * Test that the HMM filter reaches an expected level of accuracy for
   * the marginal class posterior probabilities.
   */
  @Test
  public void testMarginalClassProbabilities1() {
    final Random rng = new Random();//123452388l);

    final int S = 2;
    DefaultDataDistribution<Integer> s1Likelihood = new DefaultDataDistribution<Integer>();
    s1Likelihood.increment(0, 1d/3d);
    s1Likelihood.increment(1, 1d/3d);

    StandardHMM<Integer> trueHmm = 
        StandardHMM.create(HiddenMarkovModel.createRandom(S, s1Likelihood, rng));

    final int T = 30;
    List<SimHmmObservedValue<Integer, Integer>> samples = trueHmm.sample(rng, T);
    
    List<Integer> obsValues = Lists.newArrayList();
    for(SimHmmObservedValue<Integer, Integer> obs : samples)
      obsValues.add(obs.getObservedValue());

    StandardHMM<Integer> hmm = trueHmm.clone();

    CategoricalHmmPlFilter filter = new CategoricalHmmPlFilter(hmm, rng, false);
    
    List<RingAccumulator<MutableDouble>> classErrorRates = Lists.newArrayList();
    for (int j = 0; j < hmm.getNumStates(); j++) {
      RingAccumulator<MutableDouble> cRate = new RingAccumulator<MutableDouble>();
      classErrorRates.add(cRate);
    }

    List<Vector> forwardResults = hmm.stateBeliefs(obsValues);

    final int numIterations = 10;
    final int N = 10000;
    for (int k = 0; k < numIterations; k++) {
      filter.setNumParticles(N);
  
      CountedDataDistribution<HmmTransitionState<Integer, StandardHMM<Integer>>> distribution = 
          (CountedDataDistribution<HmmTransitionState<Integer, StandardHMM<Integer>>>) filter.createInitialLearnedObject();
      
      /*
       * Recurse through the particle filter
       */
      for (int i = 0; i < T; i++) {
        final Integer y = samples.get(i).getObservedValue();
        final ObservedValue obsState = new ObservedValue(i, y);
        filter.update(distribution, obsState);
  
        CountedDataDistribution<Integer> stateSums = new CountedDataDistribution<>(true);
        for (HmmTransitionState<Integer, StandardHMM<Integer>> state : distribution.getDomain()) {
          stateSums.adjust(state.getClassId(), distribution.getLogFraction(state), distribution.getCount(state));
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
        Doubles.toArray(errMeans), 1e-1);
  }

  @Test
  public void testBaumWelchInitialization1() {
    DefaultDataDistribution<Integer> s1Likelihood = new DefaultDataDistribution<>();
    s1Likelihood.increment(0, 1d/3d);
    s1Likelihood.increment(1, 1d/3d);

    DefaultDataDistribution<Integer> s2Likelihood = new DefaultDataDistribution<>();
    s2Likelihood.increment(0, 1d/3d);
    s2Likelihood.increment(1, 1d/3d);
    
    List<DefaultDataDistribution<Integer>> emissionFunctions = Lists.newArrayList();
    emissionFunctions.add(s1Likelihood);
    emissionFunctions.add(s2Likelihood);
    
    StandardHMM<Integer> hmm = 
        StandardHMM.create(
        new HiddenMarkovModel<>(
        VectorFactory.getDefault().copyArray(new double[] {1d/2d, 1d/2d}),
        MatrixFactory.getDefault().copyArray(new double[][] {
            // (i,j) j: from, i: to
            {1d/3d, 1d/3d}, 
            {1d/3d, 1d/3d}}),
        emissionFunctions));
    
    TreeSet<HmmTransitionState<Integer, StandardHMM<Integer>>> result = new CategoricalHmmPlFilter.
        CategoricalHmmPlUpdater(hmm, new Random()).
          expandForwardProbabilities(hmm, Lists.newArrayList(0, 1));
    
    assertTrue(result.size() == 4);
    for (HmmTransitionState<Integer, StandardHMM<Integer>> chain : result) {
      System.out.println(chain.getStateHistory());
      System.out.println("\t" + chain);
      assertEquals(Math.log(1d/4d), chain.getStateLogWeight(), 1e-7);
    }
    System.out.println(result);
  }
  
  @Test
  public void testBaumWelchInitialization2() {
    DefaultDataDistribution<Integer> s1Likelihood = new DefaultDataDistribution<>();
    s1Likelihood.increment(0, 1d/2d);
    s1Likelihood.increment(1, 1d/2d);

    DefaultDataDistribution<Integer> s2Likelihood = new DefaultDataDistribution<>();
    s2Likelihood.increment(0, 2d/3d);
    s2Likelihood.increment(1, 1d/3d);
    
    List<DefaultDataDistribution<Integer>> emissionFunctions = Lists.newArrayList();
    emissionFunctions.add(s1Likelihood);
    emissionFunctions.add(s2Likelihood);
    
    StandardHMM<Integer> hmm = 
        StandardHMM.create(
        new HiddenMarkovModel<>(
        VectorFactory.getDefault().copyArray(new double[] {1d/2d, 1d/2d}),
        MatrixFactory.getDefault().copyArray(new double[][] {
            // (i,j) j: from, i: to
            {9d/10d, 1d/10d}, 
            {1d/10d, 9d/10d}}),
        emissionFunctions));
    
    ArrayList<Integer> observations = Lists.newArrayList(0, 1);
    TreeSet<HmmTransitionState<Integer, StandardHMM<Integer>>> result = new CategoricalHmmPlFilter.
        CategoricalHmmPlUpdater(hmm, new Random()).
          expandForwardProbabilities(hmm, observations);
    
    assertTrue(result.size() == Math.pow(hmm.getNumStates(), observations.size()));
    double totalLogLikelihood = Double.NEGATIVE_INFINITY;
    Set<Double> uniqueValues = Sets.newHashSet();
    for (HmmTransitionState<Integer, StandardHMM<Integer>> chain : result) {
      System.out.println(chain.getStateHistory());
      System.out.println("\t" + chain);
      uniqueValues.add(chain.getStateLogWeight());
      totalLogLikelihood = ExtLogMath.add(totalLogLikelihood, chain.getStateLogWeight());
//      assertEquals(Math.log(1d/4d), chain.getStateLogWeight(), 1e-7);
    }
    assertEquals(0d, totalLogLikelihood, 1e-7);
    System.out.println("uniquely weighed chains = " + uniqueValues.size()
        + "/" + result.size());
  }
  
}
