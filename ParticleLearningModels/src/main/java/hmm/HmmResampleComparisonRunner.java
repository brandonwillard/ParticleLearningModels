package hmm;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.bayesian.ParticleFilter;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;
import hmm.HmmTransitionState.ResampleType;

import java.io.FileWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import utils.CountedDataDistribution;
import utils.ObservedValue;
import utils.SamplingUtils;
import au.com.bytecode.opencsv.CSVWriter;

import com.google.common.math.DoubleMath;

public class HmmResampleComparisonRunner {

  protected static final Logger log = Logger
        .getLogger(SamplingUtils.class);

  public HmmResampleComparisonRunner() {
    super();
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
  protected static <T> void waterFillResampleComparison(
      HmmPlFilter<T> wfFilter, ParticleFilter<ObservedValue<T>, ? extends HmmTransitionState<T>> rsFilter, 
        HiddenMarkovModel<T> trueHmm, final int N, final int T, 
        final int K, String outputFilename, final Random rng) throws IOException {
        
        final HiddenMarkovModel<T> hmm = trueHmm;
    
        Pair<List<T>, List<Integer>> sample = SamplingUtils.sampleWithStates(trueHmm, rng, T);
    
        wfFilter.setNumParticles(N);
        wfFilter.setResampleOnly(false);

        rsFilter.setNumParticles(N);
    
        List<Vector> forwardResults = hmm.stateBeliefs(sample.getFirst());
        ExposedHmm<T> wTrueHMM = new ExposedHmm<T>(trueHmm);
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
        String[] header = "rep,t,measurement.type,filter.type,resample.type,measurement".split(",");
        writer.writeNext(header);
    
        for (int k = 0; k < K; k++) {
          CountedDataDistribution<HmmTransitionState<T>> wfDistribution =
              ((HmmPlFilter.HmmPlUpdater<T>) wfFilter.getUpdater()).baumWelchInitialization(sample.getFirst(), N);
    
          CountedDataDistribution<HmmTransitionState<T>> rsDistribution =
              (CountedDataDistribution<HmmTransitionState<T>>) 
              rsFilter.getUpdater().createInitialParticles(N);
    
          final long numPreRuns = wfDistribution.getMaxValueKey().getTime();
          
          /*
           * Recurse through the particle filter
           */
          for (int i = 0; i < T; i++) {
      
            final double x = DoubleMath.roundToInt(sample.getSecond().get(i), RoundingMode.HALF_EVEN);
            viterbiRate.accumulate(new MutableDouble((x == viterbiResults.get(i) ? 1d : 0d)));
    
            final T y = sample.getFirst().get(i);
            final ObservedValue<T> obsState = new ObservedValue<T>(i, y);
    
            rsFilter.update(rsDistribution, obsState);
    
            /*
             * Compute and output RS forward errors
             */
            ResampleType rsResampleType = rsDistribution.getMaxValueKey().getResampleType();
            Vector rsStateProbDiffs = computeStateDiffs(i, hmm.getNumStates(), rsDistribution, forwardResults);
            String[] rsLine = {Integer.toString(k), Integer.toString(i), "p(x_t=0|y^t)", 
               rsResampleType.toString(),
               Double.toString(rsStateProbDiffs.getElement(0))};
            writer.writeNext(rsLine);
            log.info("rsStateProbDiffs=" + rsStateProbDiffs);
    
      
            if (i > numPreRuns) {
              wfFilter.update(wfDistribution, obsState);
        
              RingAccumulator<MutableDouble> pfAtTRate = new RingAccumulator<MutableDouble>();
              for (HmmTransitionState<T> state : wfDistribution.getDomain()) {
                final double err = (x == state.getState()) ? wfDistribution.getFraction(state) : 0d;
                pfAtTRate.accumulate(new MutableDouble(err));
              }
              pfRunningRate.accumulate(new MutableDouble(pfAtTRate.getSum()));
        
              ResampleType wfResampleType = wfDistribution.getMaxValueKey().getResampleType();
              Vector wfStateProbDiffs = computeStateDiffs(i, hmm.getNumStates(), wfDistribution, forwardResults);
              String[] wfLine = {Integer.toString(k), Integer.toString(i), "p(x_t=0|y^t)", "water-filling",
                 wfResampleType.toString(),
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
              for (HmmTransitionState<T> state : wfDistribution.getDomain()) {
                final WeightedValue<Integer> weighedState = state.getStateHistory().get(t);
                wfStateSums.increment(weighedState.getValue(), weighedState.getWeight());
              }
              for (HmmTransitionState<T> state : rsDistribution.getDomain()) {
                final WeightedValue<Integer> weighedState = state.getStateHistory().get(t);
                rsStateSums.increment(weighedState.getValue(), weighedState.getWeight());
              }
            } else {
              for (HmmTransitionState<T> state : wfDistribution.getDomain()) {
                wfStateSums.adjust(state.getState(), wfDistribution.getLogFraction(state), wfDistribution.getCount(state));
              }
              for (HmmTransitionState<T> state : rsDistribution.getDomain()) {
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
                  wfDistribution.getMaxValueKey().getResampleType().toString(), 
                  Double.toString(wfStateProbDiffs.getElement(0))};
            writer.writeNext(wfLine);
            String[] rsLine = {Integer.toString(k), Integer.toString(t), "p(x_t=0|y^T)", "resample", 
                rsDistribution.getMaxValueKey().getResampleType().toString(), 
                Double.toString(rsStateProbDiffs.getElement(0))};
            writer.writeNext(rsLine);
          }
        }
        writer.close();
      }

  private static <T> Vector computeStateDiffs(int t,
    int S, CountedDataDistribution<HmmTransitionState<T>> wfDistribution, List<Vector> trueDists) {
    
      CountedDataDistribution<Integer> stateSums = new CountedDataDistribution<>(true);
      for (HmmTransitionState<T> state : wfDistribution.getDomain()) {
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