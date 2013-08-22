package hmm;

import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import utils.CountedDataDistribution;
import utils.LogMath2;
import utils.SamplingUtils;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.math.DoubleMath;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.RingAverager;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

public class CategoricalHMMRunner {

  public static void main(String[] args) {
    
    DefaultDataDistribution<Integer> s1Likelihood = new DefaultDataDistribution<Integer>();
    s1Likelihood.increment(0, 1d/3d);
    s1Likelihood.increment(1, 2d/3d);

    DefaultDataDistribution<Integer> s2Likelihood = new DefaultDataDistribution<Integer>();
    s2Likelihood.increment(0, 2d/3d);
    s2Likelihood.increment(1, 1d/3d);
    
    HiddenMarkovModel<Integer> trueHmm = new HiddenMarkovModel<Integer>(
        VectorFactory.getDefault().copyArray(new double[] {1d/3d, 2d/3d}),
        MatrixFactory.getDefault().copyArray(new double[][] {
            {9d/10d, 1d/10d}, 
            {1d/10d, 9d/10d}}),
        Lists.newArrayList(s1Likelihood, s2Likelihood)
        );
    
    final Random rng = new Random();//123452388l);
    final int T = 30;
    Pair<List<Integer>, List<Integer>> sample = SamplingUtils.sampleWithStates(trueHmm, rng, T);
    
    HiddenMarkovModel<Integer> hmm = new HiddenMarkovModel<Integer>(
        VectorFactory.getDefault().copyArray(new double[] {1d/3d, 2d/3d}),
        MatrixFactory.getDefault().copyArray(new double[][] {
            {9d/10d, 1d/10d}, 
            {1d/10d, 9d/10d}}),
        Lists.newArrayList(s1Likelihood, s2Likelihood)
        );

    CategoricalHMMPLFilter filter = new CategoricalHMMPLFilter(hmm, rng);
    final int N = 100;
    filter.setNumParticles(N);

    CountedDataDistribution<HMMTransitionState<Integer>> distribution =
        filter.getUpdater().baumWelchInitialization(sample.getFirst(), N);
    final long numPreRuns = distribution.getMaxValueKey().getTime();

    
    List<Vector> forwardResults = hmm.stateBeliefs(sample.getFirst());
    ExposedHMM<Integer> wTrueHMM = new ExposedHMM<Integer>(trueHmm);
    ArrayList<Vector> b =
        wTrueHMM.computeObservationLikelihoods(sample.getFirst());
    ArrayList<WeightedValue<Vector>> alphas =
        wTrueHMM.computeForwardProbabilities(b, true);
    ArrayList<WeightedValue<Vector>> betas =
        wTrueHMM.computeBackwardProbabilities(b, alphas);
    ArrayList<Vector> gammas =
        wTrueHMM.computeStateObservationLikelihood(alphas, betas, 1d);

    List<Integer> viterbiResults = hmm.viterbi(sample.getFirst());
    
    RingAccumulator<MutableDouble> viterbiRate = new RingAccumulator<MutableDouble>();
    RingAccumulator<MutableDouble> pfRunningRate = new RingAccumulator<MutableDouble>();
    /*
     * Recurse through the particle filter
     */
    for (int i = 0; i < T; i++) {

      final double x = DoubleMath.roundToInt(sample.getSecond().get(i), RoundingMode.HALF_EVEN);
      viterbiRate.accumulate(new MutableDouble((x == viterbiResults.get(i) ? 1d : 0d)));

      if (i > numPreRuns) {
        final Integer y = sample.getFirst().get(i);
        final ObservedState obsState = new ObservedState(i, y);
        filter.update(distribution, obsState);
  
        RingAccumulator<MutableDouble> pfAtTRate = new RingAccumulator<MutableDouble>();
        CountedDataDistribution<Integer> stateSums = new CountedDataDistribution<>(true);
        for (HMMTransitionState<Integer> state : distribution.getDomain()) {
          stateSums.adjust(state.getState(), distribution.getLogFraction(state), distribution.getCount(state));
  
          final double err = (x == state.getState()) ? distribution.getFraction(state) : 0d;
          pfAtTRate.accumulate(new MutableDouble(err));
        }
        pfRunningRate.accumulate(new MutableDouble(pfAtTRate.getSum()));
  
        /*
         * Check marginal state probabilities
         */
        Preconditions.checkState(stateSums.getDomainSize() <= hmm.getNumStates());
        Vector stateProbDiffs = VectorFactory.getDefault().createVector(hmm.getNumStates());
        for (int j = 0; j < hmm.getNumStates(); j++) {
          /*
           * Sometimes all the probability goes to one class...
           */
          final double stateProb; 
          if (!stateSums.getDomain().contains(j))
            stateProb = 0d;
          else
            stateProb = stateSums.getFraction(j);
          stateProbDiffs.setElement(j, forwardResults.get(i).getElement(j) - stateProb);
        }
        System.out.println("stateProbDiffs=" + stateProbDiffs);
      }

    }
    System.out.println("viterbiRate:" + viterbiRate.getMean());
    System.out.println("pfRunningRate:" + pfRunningRate.getMean());

    RingAccumulator<MutableDouble> pfRate2 = new RingAccumulator<MutableDouble>();
    for (HMMTransitionState<Integer> state : distribution.getDomain()) {
      final double chainLogLikelihood = distribution.getLogFraction(state);
      RingAccumulator<MutableDouble> pfAtTimeRate = new RingAccumulator<MutableDouble>();
      for (int i = 0; i < T; i++) {
        final double x = DoubleMath.roundToInt(sample.getSecond().get(i), RoundingMode.HALF_EVEN);
        final double err;
        if (i < T - 1) {
          final WeightedValue<Integer> weighedState = state.getStateHistory().get(i);
          err = (x == weighedState.getValue() ? 1d : 0d); 
        } else {
          err = (x == state.getState() ? 1d : 0d); 
        }
        pfAtTimeRate.accumulate(new MutableDouble(err));
      }
      pfRate2.accumulate(new MutableDouble(pfAtTimeRate.getMean().doubleValue() 
          * Math.exp(chainLogLikelihood)));
    }

    System.out.println("pfChainRate:" + pfRate2.getSum());
  }

}
