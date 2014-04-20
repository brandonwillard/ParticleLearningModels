package plm.util.hmm;

import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.bayesian.ParticleFilter;
import gov.sandia.cognition.util.WeightedValue;

import java.io.FileWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import plm.hmm.GenericHMM;
import plm.hmm.HmmPlFilter;
import plm.hmm.HmmTransitionState;
import plm.hmm.StandardHMM;
import plm.hmm.GenericHMM.SimHmmObservedValue;
import plm.hmm.HmmPlFilter.HmmPlUpdater;
import plm.hmm.HmmTransitionState.ResampleType;
import au.com.bytecode.opencsv.CSVWriter;

import com.google.common.collect.Lists;
import com.google.common.math.DoubleMath;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;
import com.statslibextensions.util.ObservedValue;

public class HmmResampleComparisonRunner {

  protected static final Logger log = Logger
        .getLogger(HmmResampleComparisonRunner.class);

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
  protected static <H extends StandardHMM<T>, P extends HmmTransitionState<T, H>, T> void waterFillResampleComparison(
      HmmPlFilter<H, P, T> wfFilter, ParticleFilter<ObservedValue<T, Void>, P> rsFilter, 
        H trueHmm, final int N, final int T, 
        final int K, String outputFilename, final Random rng) throws IOException {
        
        final H hmm = trueHmm;
    
        List<SimHmmObservedValue<T, Integer>> samples = trueHmm.sample(rng, T);
        
        List<T> obsValues = Lists.newArrayList();
        for(SimHmmObservedValue<T, Integer> obs : samples)
          obsValues.add(obs.getObservedValue());
    
        wfFilter.setNumParticles(N);
        wfFilter.setResampleOnly(false);

        rsFilter.setNumParticles(N);
    
        List<Vector> forwardResults = hmm.stateBeliefs(obsValues);
        ArrayList<Vector> b = trueHmm.computeObservationLikelihoods(obsValues);
        ArrayList<WeightedValue<Vector>> alphas =
            trueHmm.computeForwardProbabilities(b, true);
        ArrayList<WeightedValue<Vector>> betas =
            trueHmm.computeBackwardProbabilities(b, alphas);
        ArrayList<Vector> gammas =
            trueHmm.computeStateObservationLikelihood(alphas, betas, 1d);
    
        List<Integer> viterbiResults = hmm.viterbi(obsValues);
        
        RingAccumulator<MutableDouble> viterbiRate = new RingAccumulator<MutableDouble>();
        RingAccumulator<MutableDouble> pfRunningRate = new RingAccumulator<MutableDouble>();
    
        CSVWriter writer = new CSVWriter(new FileWriter(outputFilename), ',');
        String[] header = "rep,t,measurement.type,filter.type,resample.type,measurement".split(",");
        writer.writeNext(header);
    
        for (int k = 0; k < K; k++) {
          CountedDataDistribution<P> wfDistribution =
              ((HmmPlFilter.HmmPlUpdater<H, P, T>) wfFilter.getUpdater()).baumWelchInitialization(obsValues, N);
    
          CountedDataDistribution<P> rsDistribution =
              (CountedDataDistribution<P>) 
              rsFilter.getUpdater().createInitialParticles(N);
    
          final long numPreRuns = wfDistribution.getMaxValueKey().getTime();
          
          /*
           * Recurse through the particle filter
           */
          for (int i = 0; i < T; i++) {
      
            final double x = DoubleMath.roundToInt(samples.get(i).getState(), RoundingMode.HALF_EVEN);
            viterbiRate.accumulate(new MutableDouble((x == viterbiResults.get(i) ? 1d : 0d)));
    
            final T y = samples.get(i).getObservedValue();
            final ObservedValue<T, Void> obsState = ObservedValue.create(i, y);
    
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
              for (P state : wfDistribution.getDomain()) {
                final double err = (x == state.getClassId()) ? wfDistribution.getFraction(state) : 0d;
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
    
            CountedDataDistribution<Integer> wfStateSums = new CountedDataDistribution<Integer>(true);
            CountedDataDistribution<Integer> rsStateSums = new CountedDataDistribution<Integer>(true);
            if (t < T - 1) {
              for (HmmTransitionState<T, H> state : wfDistribution.getDomain()) {
                final WeightedValue<Integer> weighedState = state.getStateHistory().get(t);
                wfStateSums.increment(weighedState.getValue(), weighedState.getWeight());
              }
              for (HmmTransitionState<T, H> state : rsDistribution.getDomain()) {
                final WeightedValue<Integer> weighedState = state.getStateHistory().get(t);
                rsStateSums.increment(weighedState.getValue(), weighedState.getWeight());
              }
            } else {
              for (P state : wfDistribution.getDomain()) {
                wfStateSums.adjust(state.getClassId(), wfDistribution.getLogFraction(state), wfDistribution.getCount(state));
              }
              for (P state : rsDistribution.getDomain()) {
                rsStateSums.adjust(state.getClassId(), rsDistribution.getLogFraction(state), rsDistribution.getCount(state));
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

  private static <H extends StandardHMM<T>, P extends HmmTransitionState<T, H>, T> Vector computeStateDiffs(int t,
    int S, CountedDataDistribution<P> wfDistribution, List<Vector> trueDists) {
    
      CountedDataDistribution<Integer> stateSums = new CountedDataDistribution<Integer>(true);
      for (P state : wfDistribution.getDomain()) {
        stateSums.adjust(state.getClassId(), wfDistribution.getLogFraction(state), wfDistribution.getCount(state));
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