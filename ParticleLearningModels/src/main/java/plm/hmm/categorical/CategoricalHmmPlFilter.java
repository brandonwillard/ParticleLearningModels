package plm.hmm.categorical;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
import gov.sandia.cognition.util.WeightedValue;

import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import org.apache.log4j.Logger;
import org.paukov.combinatorics.Factory;
import org.paukov.combinatorics.Generator;
import org.paukov.combinatorics.ICombinatoricsVector;

import plm.hmm.GenericHMM;
import plm.hmm.HmmPlFilter;
import plm.hmm.HmmTransitionState;
import plm.hmm.StandardHMM;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.math.DoubleMath;
import com.statslibextensions.math.ExtLogMath;
import com.statslibextensions.statistics.CountedDataDistribution;
import com.statslibextensions.statistics.ExtSamplingUtils;
import com.statslibextensions.statistics.distribution.WFCountedDataDistribution;
import com.statslibextensions.util.ObservedValue;

/**
 * A particle filter for categorical response HMMs that provides the option of
 * water-filling resampling the expanded step-forward state space.
 * 
 * @author bwillard
 * 
 */
public class CategoricalHmmPlFilter extends HmmPlFilter<StandardHMM<Integer>, HmmTransitionState<Integer, StandardHMM<Integer>>, Integer> {

  public static class CategoricalHmmPlUpdater extends HmmPlUpdater<StandardHMM<Integer>, HmmTransitionState<Integer, StandardHMM<Integer>>, Integer> {

    private static final long serialVersionUID = 7961478795131339665L;

    public CategoricalHmmPlUpdater(StandardHMM<Integer> hmm,
      Random rng) {
      super(hmm, rng);
    }

    @Override
    public DataDistribution<HmmTransitionState<Integer, StandardHMM<Integer>>>
        createInitialParticles(int numParticles) {
      final CountedDataDistribution<HmmTransitionState<Integer, StandardHMM<Integer>>> initialParticles =
          new CountedDataDistribution<HmmTransitionState<Integer, StandardHMM<Integer>>>(numParticles, true);
      for (int i = 0; i < numParticles; i++) {
        final int sampledState =
            DiscreteSamplingUtil.sampleIndexFromProbabilities(
                this.rng, this.priorHmm.getClassMarginalProbabilities());
        final HmmTransitionState<Integer, StandardHMM<Integer>> particle =
            new HmmTransitionState<Integer, StandardHMM<Integer>>(this.priorHmm, sampledState,
                null);

        final double logWeight = -Math.log(numParticles);
        particle.setStateLogWeight(logWeight);
        initialParticles.increment(particle, logWeight);
      }
      return initialParticles;
    }

    @Override
    public WFCountedDataDistribution<HmmTransitionState<Integer, StandardHMM<Integer>>>
        baumWelchInitialization(List<Integer> sample,
          final int numParticles) {

      final int numPreRuns =
          DoubleMath.roundToInt(
              Math.log(numParticles) / Math.log(priorHmm.getNumStates()),
              RoundingMode.CEILING);
      final TreeSet<HmmTransitionState<Integer, StandardHMM<Integer>>> expandedStates =
          this.expandForwardProbabilities(priorHmm,
              sample.subList(0, numPreRuns));
      final Iterator<HmmTransitionState<Integer, StandardHMM<Integer>>> descIter =
          expandedStates.descendingIterator();
      final Set<Double> uniqueWeights = Sets.newHashSet();
      final double[] logWeights = new double[expandedStates.size()];
      double totalLogWeight = Double.NEGATIVE_INFINITY;
      final List<HmmTransitionState<Integer, StandardHMM<Integer>>> domain =
          Lists.newArrayList();
      for (int i = 0; i < expandedStates.size(); i++) {
        final HmmTransitionState<Integer, StandardHMM<Integer>> state = descIter.next();
        uniqueWeights.add(state.getStateLogWeight());
        logWeights[i] = state.getStateLogWeight();
        totalLogWeight =
            ExtLogMath.add(totalLogWeight, state.getStateLogWeight());
        domain.add(state);
      }

      /*
       * Now, water-fill these results.
       */
      final WFCountedDataDistribution<HmmTransitionState<Integer, StandardHMM<Integer>>> distribution =
          ExtSamplingUtils.waterFillingResample(logWeights,
              totalLogWeight, domain, rng, numParticles);

      System.out.println("unique weights = " + uniqueWeights.size());

      return distribution;
    }
    
    protected <T> List<Vector> computeSmoothedJointProbs(final StandardHMM<T> hmm, List<T> observations) {
  
      /*
       * Compute Baum-Welch smoothed distribution
       */
      final ArrayList<Vector> obsLikelihoodSequence =
          hmm.computeObservationLikelihoods(observations);
      final ArrayList<WeightedValue<Vector>> forwardProbabilities =
          hmm.computeForwardProbabilities(obsLikelihoodSequence, true);
  
      final List<Vector> jointProbs = Lists.newArrayList();
      for (int i = 0; i < forwardProbabilities.size() - 1; i++) {
        final WeightedValue<Vector> input = forwardProbabilities.get(i);
        final Vector prod =
            hmm.getTransitionProbability().times(input.getValue());
        jointProbs.add(prod.scale(1d / prod.norm1()));
      }
      jointProbs
          .add(Iterables.getLast(forwardProbabilities).getValue());
      
      return jointProbs;
    }

    /**
     * Computes the joint smoothed probability over the given observations 
     * until numParticles-many paths are reached,
     * then returns the resulting ordered, weighed sample path.
     * 
     * @param hmm
     * @param observations
     * @param numParticles
     * @return
     */
    public <T> TreeSet<HmmTransitionState<T, StandardHMM<T>>>
        expandForwardProbabilities(final StandardHMM<T> hmm,
          List<T> observations) {

      List<Vector> jointProbs = computeSmoothedJointProbs(hmm, observations);
  
      final Integer[] states = new Integer[hmm.getNumStates()];
      for (int i = 0; i < hmm.getNumStates(); i++) {
        states[i] = i;
      }
      final ICombinatoricsVector<Integer> initialVector =
          Factory.createVector(states);
      final Generator<Integer> gen =
          Factory.createPermutationWithRepetitionGenerator(
              initialVector, observations.size());
  
      /*
       * Iterate through possible state permutations and find their
       * log likelihoods via the Baum-Welch results above.
       */
      final TreeSet<HmmTransitionState<T, StandardHMM<T>>> orderedDistribution =
          Sets.newTreeSet(new Comparator<HmmTransitionState<T, StandardHMM<T>>>() {
            @Override
            public int compare(HmmTransitionState<T, StandardHMM<T>> o1,
              HmmTransitionState<T, StandardHMM<T>> o2) {
              final int compVal =
                  Double.compare(o1.getStateLogWeight(),
                      o2.getStateLogWeight());
              return compVal == 0 ? 1 : compVal;
            }
          });
      for (final ICombinatoricsVector<Integer> combination : gen) {
  
        HmmTransitionState<T, StandardHMM<T>> currentState = null;
        double logWeightOfState = 0d;
        for (int i = 0; i < combination.getSize(); i++) {
          final Vector smoothedProbsAtTime =
              Iterables.get(jointProbs, i);
          final int stateAtTime = combination.getVector().get(i);
          // TODO assuming it's normalized, is that true?
          final double logWeightAtTime =
              Math.log(smoothedProbsAtTime.getElement(stateAtTime));
  
          logWeightOfState += logWeightAtTime;
          if (currentState == null) {
            currentState =
                new HmmTransitionState<T, StandardHMM<T>>(hmm, stateAtTime, 
                    new ObservedValue<T, Void>(i, observations.get(i)));
          } else {
            currentState =
                new HmmTransitionState<T, StandardHMM<T>>(currentState, currentState.getHmm(), stateAtTime, 
                    new ObservedValue<T, Void>(i, observations.get(i)));
          }
          currentState.setStateLogWeight(logWeightOfState);
        }
        orderedDistribution.add(currentState);
      }
  
      return orderedDistribution;
    }

    @Override
    public double computeLogLikelihood(
      HmmTransitionState<Integer, StandardHMM<Integer>> particle,
      ObservedValue<Integer, Void> observation) {
      final ComputableDistribution<Integer> f =
          particle.getHmm().getEmissionFunction(null, particle.getClassId());
      return f.getProbabilityFunction().logEvaluate(
          observation.getObservedValue());
    }

    @Override
    public HmmTransitionState<Integer, StandardHMM<Integer>> update(
      HmmTransitionState<Integer, StandardHMM<Integer>> previousParameter) {
      return previousParameter.clone();
    }

  }

  private static final long serialVersionUID = -7387680621521036135L;

  final Logger log = Logger.getLogger(CategoricalHmmPlFilter.class);

  public CategoricalHmmPlFilter(StandardHMM<Integer> hmm,
    Random rng, boolean resampleOnly) {
    super(resampleOnly);
    this.setUpdater(new CategoricalHmmPlUpdater(hmm, rng));
    this.setRandom(rng);
    this.resampleOnly = resampleOnly;
  }

  @Override
  public void update(
    DataDistribution<HmmTransitionState<Integer, StandardHMM<Integer>>> target,
    ObservedValue<Integer, Void> data) {
    super.update(target, data);
  }

  @Override
  protected HmmTransitionState<Integer, StandardHMM<Integer>> propagate(
      HmmTransitionState<Integer, StandardHMM<Integer>> particle, int newClassId, ObservedValue<Integer, Void> data) {
    return new HmmTransitionState<Integer, StandardHMM<Integer>>(particle, particle.getHmm(), newClassId,
                data);
  }



}
