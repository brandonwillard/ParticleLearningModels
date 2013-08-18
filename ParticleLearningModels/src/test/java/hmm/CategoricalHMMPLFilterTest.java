package hmm;

import static org.junit.Assert.*;
import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.util.Pair;

import java.math.RoundingMode;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import utils.CountedDataDistribution;
import utils.SamplingUtils;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.math.DoubleMath;

public class CategoricalHMMPLFilterTest {

  /**
   * Test that the HMM filter reaches an expected level of accuracy for
   * the marginal class posterior probabilities.
   */
  @Test
  public void testMarginalClassProbabilities() {
    DefaultDataDistribution<Double> s1Likelihood = new DefaultDataDistribution<Double>();
    s1Likelihood.increment(0d, 0.5);
    s1Likelihood.increment(1d, 0.5);

    DefaultDataDistribution<Double> s2Likelihood = new DefaultDataDistribution<Double>();
    s2Likelihood.increment(0d, 1d/3d);
    s2Likelihood.increment(1d, 2d/3d);
    
    HiddenMarkovModel<Double> trueHmm = new HiddenMarkovModel<Double>(
        VectorFactory.getDefault().copyArray(new double[] {1d/3d, 2d/3d}),
        MatrixFactory.getDefault().copyArray(new double[][] {{0.5, 0.5}, {0.5, 0.5}}),
        Lists.newArrayList(s1Likelihood, s2Likelihood)
        );
    
    final Random rng = new Random();//123452388l);
    final int T = 3;
    Pair<List<Double>, List<Integer>> sample = SamplingUtils.sampleWithStates(trueHmm, rng, T);
    
    HiddenMarkovModel<Double> hmm = new HiddenMarkovModel<Double>(
        VectorFactory.getDefault().copyArray(new double[] {0.5, 0.5}),
        MatrixFactory.getDefault().copyArray(new double[][] {{0.5, 0.5}, {0.5, 0.5}}),
        Lists.newArrayList(s1Likelihood, s2Likelihood)
        );

    CategoricalHMMPLFilter filter = new CategoricalHMMPLFilter(hmm, rng);
    
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
  
      CountedDataDistribution<HMMTransitionState<Double>> distribution = 
          (CountedDataDistribution<HMMTransitionState<Double>>) filter.createInitialLearnedObject();
      
      /*
       * Recurse through the particle filter
       */
      for (int i = 0; i < T; i++) {
        final Double y = sample.getFirst().get(i);
        filter.update(distribution, y);
  
        CountedDataDistribution<Integer> stateSums = new CountedDataDistribution<>(true);
        for (HMMTransitionState<Double> state : distribution.getDomain()) {
          stateSums.adjust(state.getState(), distribution.getLogFraction(state), distribution.getCount(state));
        }
  
        /*
         * Check marginal state probabilities
         */
        Preconditions.checkState(stateSums.getDomainSize() <= hmm.getNumStates());
        for (int j = 0; j < hmm.getNumStates(); j++) {
          final double classError = forwardResults.get(i).getElement(j) - stateSums.getFraction(j);
          classErrorRates.get(j).accumulate(new MutableDouble(classError));
        }
      }
    }
    
    for (int j = 0; j < hmm.getNumStates(); j++) {
      System.out.println("meanClassProbError=" + classErrorRates.get(j).getMean().value);
      assertEquals(0d, classErrorRates.get(j).getMean().value, 5e-3);
    }
  }

}
