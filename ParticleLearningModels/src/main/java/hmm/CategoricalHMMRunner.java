package hmm;

import java.io.FileWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import org.apache.log4j.Logger;

import utils.CountedDataDistribution;
import utils.LogMath2;
import utils.SamplingUtils;
import au.com.bytecode.opencsv.CSVWriter;

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

  final static Logger log = Logger
      .getLogger(SamplingUtils.class);

  public static void main(String[] args) throws IOException {

    final long seed = new Random().nextLong();
    log.info("seed=" + seed);

    DefaultDataDistribution<Integer> s1Likelihood = new DefaultDataDistribution<Integer>();
    s1Likelihood.increment(0, 1d/2d);
    s1Likelihood.increment(1, 1d/2d);

    DefaultDataDistribution<Integer> s2Likelihood = s1Likelihood;
    
    HiddenMarkovModel<Integer> trueHmm1 = new HiddenMarkovModel<Integer>(
        VectorFactory.getDefault().copyArray(new double[] {1d/2d, 1d/2d}),
        MatrixFactory.getDefault().copyArray(new double[][] {
            {1d/2d, 1d/2d}, 
            {1d/2d, 1d/2d}}),
        Lists.newArrayList(s1Likelihood, s2Likelihood)
        );
    waterFillResampleComparison(trueHmm1, 10000, 30, 10, "hmm-wf-rs-10000-class-errors-m1.csv", seed);

    HiddenMarkovModel<Integer> trueHmm2 = new HiddenMarkovModel<Integer>(
        VectorFactory.getDefault().copyArray(new double[] {1d/3d, 2d/3d}),
        MatrixFactory.getDefault().copyArray(new double[][] {
            {9d/10d, 1d/10d}, 
            {1d/10d, 9d/10d}}),
        Lists.newArrayList(s1Likelihood, s2Likelihood)
        );
    waterFillResampleComparison(trueHmm2, 10000, 30, 10, "hmm-wf-rs-10000-class-errors-m2.csv", seed);
  }
    
  /**
   * Compute class probability errors for N particles, series length T and K repetitions under
   * a resample-only and water-filling HMM specified by trueHmm.<br>
   * Results are output into outputFilename.
   * 
   * @param trueHmm
   * @param N
   * @param T
   * @param K
   * @param outputFilename
   * @param seed
   * @throws IOException 
   */
  private static void waterFillResampleComparison(HiddenMarkovModel<Integer> trueHmm, final int N, 
    final int T, final int K, String outputFilename, final Long seed) throws IOException {
    
    final HiddenMarkovModel<Integer> hmm = trueHmm;

    final Random rng = seed != null ? new Random(seed) : new Random();
    Pair<List<Integer>, List<Integer>> sample = SamplingUtils.sampleWithStates(trueHmm, rng, T);

    CategoricalHMMPLFilter wfFilter = new CategoricalHMMPLFilter(hmm, rng, false);
    wfFilter.setNumParticles(N);
    CategoricalHMMPLFilter rsFilter = new CategoricalHMMPLFilter(hmm, rng, true);
    rsFilter.setNumParticles(N);

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

    CSVWriter writer = new CSVWriter(new FileWriter(outputFilename), ',');
    String[] header = "rep,t,type,filter,wf.resampled,error".split(",");
    writer.writeNext(header);

    for (int k = 0; k < K; k++) {
      CountedDataDistribution<HMMTransitionState<Integer>> wfDistribution =
          wfFilter.getUpdater().baumWelchInitialization(sample.getFirst(), N);

      CountedDataDistribution<HMMTransitionState<Integer>> rsDistribution =
          (CountedDataDistribution<HMMTransitionState<Integer>>) 
          rsFilter.getUpdater().createInitialParticles(N);

      final long numPreRuns = wfDistribution.getMaxValueKey().getTime();
      
      /*
       * Recurse through the particle filter
       */
      for (int i = 0; i < T; i++) {
  
        final double x = DoubleMath.roundToInt(sample.getSecond().get(i), RoundingMode.HALF_EVEN);
        viterbiRate.accumulate(new MutableDouble((x == viterbiResults.get(i) ? 1d : 0d)));

        final Integer y = sample.getFirst().get(i);
        final ObservedState obsState = new ObservedState(i, y);

        rsFilter.update(rsDistribution, obsState);

        /*
         * Compute and output RS forward errors
         */
        Vector rsStateProbDiffs = computeStateDiffs(i, hmm.getNumStates(), rsDistribution, forwardResults);
        String[] rsLine = {Integer.toString(k), Integer.toString(i), "p(x_t=0|y^t)", "resample",
           "FALSE", Double.toString(rsStateProbDiffs.getElement(0))};
        writer.writeNext(rsLine);
        log.info("rsStateProbDiffs=" + rsStateProbDiffs);

  
        if (i > numPreRuns) {
          wfFilter.update(wfDistribution, obsState);
    
          RingAccumulator<MutableDouble> pfAtTRate = new RingAccumulator<MutableDouble>();
          boolean wasWaterFilled = false;
          for (HMMTransitionState<Integer> state : wfDistribution.getDomain()) {
            wasWaterFilled = state.wasWaterFillingApplied();
            final double err = (x == state.getState()) ? wfDistribution.getFraction(state) : 0d;
            pfAtTRate.accumulate(new MutableDouble(err));
          }
          pfRunningRate.accumulate(new MutableDouble(pfAtTRate.getSum()));
    
          Vector wfStateProbDiffs = computeStateDiffs(i, hmm.getNumStates(), wfDistribution, forwardResults);
          String[] wfLine = {Integer.toString(k), Integer.toString(i), "p(x_t=0|y^t)", "water-filling",
             (wasWaterFilled ? "TRUE" : "FALSE"),
             Double.toString(wfStateProbDiffs.getElement(0))};
          writer.writeNext(wfLine);
          log.info("wfStateProbDiffs=" + wfStateProbDiffs);

        }
  
      }

      log.info("viterbiRate:" + viterbiRate.getMean());
      log.info("pfRunningRate:" + pfRunningRate.getMean());
  
//      RingAccumulator<MutableDouble> pfRate2 = new RingAccumulator<MutableDouble>();
//      for (HMMTransitionState<Integer> state : distribution.getDomain()) {
//        final double chainLogLikelihood = distribution.getLogFraction(state);
//        RingAccumulator<MutableDouble> pfAtTimeRate = new RingAccumulator<MutableDouble>();
//        for (int i = 0; i < T; i++) {
//          final double x = DoubleMath.roundToInt(sample.getSecond().get(i), RoundingMode.HALF_EVEN);
//          final double err;
//          if (i < T - 1) {
//            final WeightedValue<Integer> weighedState = state.getStateHistory().get(i);
//            err = (x == weighedState.getValue() ? 1d : 0d); 
//          } else {
//            err = (x == state.getState() ? 1d : 0d); 
//          }
//          pfAtTimeRate.accumulate(new MutableDouble(err));
//        }
//        pfRate2.accumulate(new MutableDouble(pfAtTimeRate.getMean().doubleValue() 
//            * Math.exp(chainLogLikelihood)));
//      }
//      log.info("pfChainRate:" + pfRate2.getSum());

      /*
       * Loop through the smoothed trajectories and compute the
       * class probabilities for each state.
       */
      for (int t = 0; t < T; t++) {

        CountedDataDistribution<Integer> wfStateSums = new CountedDataDistribution<>(true);
        CountedDataDistribution<Integer> rsStateSums = new CountedDataDistribution<>(true);
        if (t < T - 1) {
          for (HMMTransitionState<Integer> state : wfDistribution.getDomain()) {
            final WeightedValue<Integer> weighedState = state.getStateHistory().get(t);
            wfStateSums.increment(weighedState.getValue(), weighedState.getWeight());
          }
          for (HMMTransitionState<Integer> state : rsDistribution.getDomain()) {
            final WeightedValue<Integer> weighedState = state.getStateHistory().get(t);
            rsStateSums.increment(weighedState.getValue(), weighedState.getWeight());
          }
        } else {
          for (HMMTransitionState<Integer> state : wfDistribution.getDomain()) {
            wfStateSums.adjust(state.getState(), wfDistribution.getLogFraction(state), wfDistribution.getCount(state));
          }
          for (HMMTransitionState<Integer> state : rsDistribution.getDomain()) {
            rsStateSums.adjust(state.getState(), rsDistribution.getLogFraction(state), rsDistribution.getCount(state));
          }
        }

        Vector wfStateProbDiffs = VectorFactory.getDefault().createVector(hmm.getNumStates());
        Vector rsStateProbDiffs = VectorFactory.getDefault().createVector(hmm.getNumStates());
        for (int j = 0; j < hmm.getNumStates(); j++) {
          /*
           * Sometimes all the probability goes to one class...
           */
          final double wfStateProb; 
          if (!wfStateSums.getDomain().contains(j))
            wfStateProb = 0d;
          else
            wfStateProb = wfStateSums.getFraction(j);
          wfStateProbDiffs.setElement(j, gammas.get(t).getElement(j) - wfStateProb);

          final double rsStateProb; 
          if (!rsStateSums.getDomain().contains(j))
            rsStateProb = 0d;
          else
            rsStateProb = rsStateSums.getFraction(j);
          rsStateProbDiffs.setElement(j, gammas.get(t).getElement(j) - rsStateProb);
        }
        String[] wfLine = {Integer.toString(k), Integer.toString(t), "p(x_t=0|y^T)", "water-filling",
            "FALSE", Double.toString(wfStateProbDiffs.getElement(0))};
        writer.writeNext(wfLine);
        String[] rsLine = {Integer.toString(k), Integer.toString(t), "p(x_t=0|y^T)", "resample", 
            "FALSE", Double.toString(rsStateProbDiffs.getElement(0))};
        writer.writeNext(rsLine);
      }
    }
    writer.close();
  }

  private static <T> Vector computeStateDiffs(int t, int S, 
    CountedDataDistribution<HMMTransitionState<T>> wfDistribution, List<Vector> trueDists) {

    CountedDataDistribution<Integer> stateSums = new CountedDataDistribution<>(true);
    for (HMMTransitionState<T> state : wfDistribution.getDomain()) {
      stateSums.adjust(state.getState(), wfDistribution.getLogFraction(state), wfDistribution.getCount(state));
    }
    
    Vector stateProbDiffs = VectorFactory.getDefault().createVector(S);
    for (int j = 0; j < S; j++) {
      /*
       * Sometimes all the probability goes to one class...
       */
      final double stateProb; 
      if (!stateSums.getDomain().contains(j))
        stateProb = 0d;
      else
        stateProb = stateSums.getFraction(j);
      stateProbDiffs.setElement(j, trueDists.get(t).getElement(j) - stateProb);
    }

    return stateProbDiffs;
  }
}
