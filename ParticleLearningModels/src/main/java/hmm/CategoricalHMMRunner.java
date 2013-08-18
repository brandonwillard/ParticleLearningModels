package hmm;

import java.math.RoundingMode;
import java.util.List;
import java.util.Map;
import java.util.Random;

import utils.CountedDataDistribution;
import utils.LogMath2;
import utils.SamplingUtils;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
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
    final int N = 100000;
    filter.setNumParticles(N);

    CountedDataDistribution<HMMTransitionState<Double>> distribution = 
        (CountedDataDistribution<HMMTransitionState<Double>>) filter.createInitialLearnedObject();
    
    List<Vector> forwardResults = hmm.stateBeliefs(sample.getFirst());
    List<Integer> viterbiResults = hmm.viterbi(sample.getFirst());
    
    RingAccumulator<MutableDouble> viterbiRate = new RingAccumulator<MutableDouble>();
    RingAccumulator<MutableDouble> pfRunningRate = new RingAccumulator<MutableDouble>();
    /*
     * Recurse through the particle filter
     */
    for (int i = 0; i < T; i++) {
      final Double y = sample.getFirst().get(i);
      filter.update(distribution, y);

      final double x = DoubleMath.roundToInt(sample.getSecond().get(i), RoundingMode.HALF_EVEN);
      viterbiRate.accumulate(new MutableDouble(
          (x == viterbiResults.get(i) ? 1d : 0d)
          ));
      RingAccumulator<MutableDouble> pfAtTRate = new RingAccumulator<MutableDouble>();
      CountedDataDistribution<Integer> stateSums = new CountedDataDistribution<>(true);
      for (HMMTransitionState<Double> state : distribution.getDomain()) {
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
        stateProbDiffs.setElement(j, forwardResults.get(i).getElement(j) - 
            stateSums.getFraction(j));
      }
      System.out.println("stateProbDiffs=" + stateProbDiffs);

    }
    System.out.println("viterbiRate:" + viterbiRate.getMean());
    System.out.println("pfRunningRate:" + pfRunningRate.getMean());

    /*
     * Compute the error rates for the final particles over their realized trajectories
     */
    RingAccumulator<MutableDouble> pfRate = new RingAccumulator<MutableDouble>();
    for (int i = 0; i < T; i++) {
      /*
       * We need to normalize these weights first..
       */
      double logWeightSum = Double.NEGATIVE_INFINITY;
      for (HMMTransitionState<Double> state : distribution.getDomain()) {
        final WeightedValue<Integer> weighedState = state.getStateHistory().get(i);
        final double logWeight = weighedState.getWeight();
        logWeightSum = LogMath2.add(logWeightSum, logWeight);
      }

      RingAccumulator<MutableDouble> pfAtTimeRate = new RingAccumulator<MutableDouble>();
      for (HMMTransitionState<Double> state : distribution.getDomain()) {
        final WeightedValue<Integer> weighedState = state.getStateHistory().get(i);
        final double x = DoubleMath.roundToInt(sample.getSecond().get(i), RoundingMode.HALF_EVEN);
        final double logWeight = weighedState.getWeight();
        final double err = (x == weighedState.getValue()) ? Math.exp(logWeight - logWeightSum) : 0d;
        pfAtTimeRate.accumulate(new MutableDouble(err));
      }

      pfRate.accumulate(new MutableDouble(pfAtTimeRate.getSum()));
    }
    System.out.println("pfRate:" + pfRate.getMean());

    RingAccumulator<MutableDouble> pfRate2 = new RingAccumulator<MutableDouble>();
    for (HMMTransitionState<Double> state : distribution.getDomain()) {
      RingAccumulator<MutableDouble> pfAtTimeRate = new RingAccumulator<MutableDouble>();
      /*
       * We need to normalize these weights first..
       */
      double logWeightSum = Double.NEGATIVE_INFINITY;
      for (int i = 0; i < T; i++) {
        final WeightedValue<Integer> weighedState = state.getStateHistory().get(i);
        final double logWeight = weighedState.getWeight();
        logWeightSum = LogMath2.add(logWeightSum, logWeight);
      }
      for (int i = 0; i < T; i++) {
        final WeightedValue<Integer> weighedState = state.getStateHistory().get(i);
        final double x = DoubleMath.roundToInt(sample.getSecond().get(i), RoundingMode.HALF_EVEN);
        final double logWeight = weighedState.getWeight();
        final double err = (x == weighedState.getValue() ? 1d : 0d) * Math.exp(logWeight - logWeightSum);
        pfAtTimeRate.accumulate(new MutableDouble(err));
      }
      pfRate2.accumulate(new MutableDouble(pfAtTimeRate.getSum()));
    }

    System.out.println("pfRate2:" + pfRate2.getMean());
  }

}
