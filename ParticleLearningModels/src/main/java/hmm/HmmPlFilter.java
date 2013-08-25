package hmm;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
import gov.sandia.cognition.statistics.bayesian.AbstractParticleFilter;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.WeightedValue;

import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import org.paukov.combinatorics.Factory;
import org.paukov.combinatorics.Generator;
import org.paukov.combinatorics.ICombinatoricsVector;

import utils.CountedDataDistribution;
import utils.LogMath2;
import utils.MutableDoubleCount;
import utils.ObservedValue;
import utils.SamplingUtils;
import utils.WFCountedDataDistribution;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.math.DoubleMath;
import com.google.common.primitives.Doubles;

public abstract class HmmPlFilter<Response>
    extends
    AbstractParticleFilter<ObservedValue<Response>, HmmTransitionState<Response>> {

  public static abstract class HmmPlUpdater<Response> extends
      AbstractCloneableSerializable implements
      Updater<ObservedValue<Response>, HmmTransitionState<Response>> {

    private static final long serialVersionUID = 1675005722404209890L;

    final protected HiddenMarkovModel<Response> hmm;
    final protected Random rng;

    public HmmPlUpdater(HiddenMarkovModel<Response> hmm, Random rng) {
      this.hmm = hmm;
      this.rng = rng;
    }

    public WFCountedDataDistribution<HmmTransitionState<Response>>
        baumWelchInitialization(List<Response> sample,
          final int numParticles) {

      final int numPreRuns =
          DoubleMath.roundToInt(
              Math.log(numParticles) / Math.log(hmm.getNumStates()),
              RoundingMode.CEILING);
      final TreeSet<HmmTransitionState<Response>> expandedStates =
          HmmPlFilter.expandForwardProbabilities(hmm,
              sample.subList(0, numPreRuns));
      final Iterator<HmmTransitionState<Response>> descIter =
          expandedStates.descendingIterator();
      final Set<Double> uniqueWeights = Sets.newHashSet();
      final double[] logWeights = new double[expandedStates.size()];
      double totalLogWeight = Double.NEGATIVE_INFINITY;
      final List<HmmTransitionState<Response>> domain =
          Lists.newArrayList();
      for (int i = 0; i < expandedStates.size(); i++) {
        final HmmTransitionState<Response> state = descIter.next();
        uniqueWeights.add(state.getStateLogWeight());
        logWeights[i] = state.getStateLogWeight();
        totalLogWeight =
            LogMath2.add(totalLogWeight, state.getStateLogWeight());
        domain.add(state);
      }

      /*
       * Now, water-fill these results.
       */
      final WFCountedDataDistribution<HmmTransitionState<Response>> distribution =
          SamplingUtils.waterFillingResample(logWeights,
              totalLogWeight, domain, rng, numParticles);

      System.out.println("unique weights = " + uniqueWeights.size());

      return distribution;
    }

    @Override
    public double computeLogLikelihood(
      HmmTransitionState<Response> particle,
      ObservedValue<Response> observation) {
      return Double.NaN;
    }

    @Override
    public DataDistribution<HmmTransitionState<Response>>
        createInitialParticles(int numParticles) {
      final CountedDataDistribution<HmmTransitionState<Response>> initialParticles =
          new CountedDataDistribution<>(numParticles, true);
      for (int i = 0; i < numParticles; i++) {
        final int sampledState =
            DiscreteSamplingUtil.sampleIndexFromProbabilities(
                this.rng, this.hmm.getInitialProbability());
        final HmmTransitionState<Response> particle =
            new HmmTransitionState<Response>(this.hmm, sampledState,
                0l);

        final double logWeight = -Math.log(numParticles);
        particle.setStateLogWeight(logWeight);
        initialParticles.increment(particle, logWeight);
      }
      return initialParticles;
    }

    @Override
    public HmmTransitionState<Response> update(
      HmmTransitionState<Response> previousParameter) {
      return previousParameter.clone();
    }
  }

  private static final long serialVersionUID = 2271089378484039661L;

  protected boolean resampleOnly;

  public HmmPlFilter(boolean resampleOnly) {
    super();
    this.resampleOnly = resampleOnly;
  }

  public boolean isResampleOnly() {
    return resampleOnly;
  }

  public void setResampleOnly(boolean resampleOnly) {
    this.resampleOnly = resampleOnly;
  }

  @Override
  public void update(
    DataDistribution<HmmTransitionState<Response>> target,
    ObservedValue<Response> data) {

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final List<Double> logLikelihoods = Lists.newArrayList();
    final List<HmmTransitionState<Response>> particleSupport =
        Lists.newArrayList();
    for (final HmmTransitionState<Response> particle : target
        .getDomain()) {
      final HiddenMarkovModel<Response> hmm = particle.getHmm();

      final int particleCount =
          ((CountedDataDistribution) target).getCount(particle);
      int i = 0;

      final double particlePriorLogLik =
          target.getLogFraction(particle);
      for (final ComputableDistribution<Response> f : particle
          .getHmm().getEmissionFunctions()) {

        final HmmTransitionState<Response> transState =
            new HmmTransitionState<Response>(particle, i,
                data.getTime());

        final double transStateLogLik =
            this.updater.computeLogLikelihood(transState, data)
                + particlePriorLogLik
                + Math.log(hmm.getTransitionProbability().getElement(
                    i, particle.getState()));

        logLikelihoods.addAll(Collections.nCopies(particleCount,
            transStateLogLik));
        particleSupport.add(transState);
        if (particleCount - 1 > 0) {
          particleSupport.addAll(Collections.nCopies(
              particleCount - 1, transState.clone()));
          //            for (int k = 0; k < particleCount-1; k++) { 
          //              particleSupport.add(transState.clone());
          //            }
        }

        particleTotalLogLikelihood =
            LogMath2.add(particleTotalLogLikelihood, transStateLogLik
                + Math.log(particleCount));
        i++;
      }
    }

    final boolean wasWaterFillingApplied;
    final CountedDataDistribution<HmmTransitionState<Response>> resampledParticles;
    if (this.resampleOnly) {
      resampledParticles = new CountedDataDistribution<>(true);
      resampledParticles.incrementAll(SamplingUtils
          .sampleMultipleLogScale(
              SamplingUtils.accumulate(logLikelihoods),
              particleTotalLogLikelihood, particleSupport,
              this.random, this.numParticles, true));
      wasWaterFillingApplied = false;
    } else {
      /*
       * Water-filling resample, for a smoothed predictive set
       */
      resampledParticles =
          SamplingUtils.waterFillingResample(
              Doubles.toArray(logLikelihoods),
              particleTotalLogLikelihood, particleSupport,
              this.random, this.numParticles);
      wasWaterFillingApplied =
          ((WFCountedDataDistribution) resampledParticles)
              .wasWaterFillingApplied();
    }

    /*
     * Propagate
     */
    final CountedDataDistribution<HmmTransitionState<Response>> updatedDist =
        new CountedDataDistribution<>(true);
    for (final Entry<HmmTransitionState<Response>, MutableDouble> entry : resampledParticles
        .asMap().entrySet()) {
      final HmmTransitionState<Response> updatedEntry =
          this.updater.update(entry.getKey());
      updatedEntry.setWasWaterFillingApplied(wasWaterFillingApplied);
      updatedEntry.setStateLogWeight(entry.getValue().doubleValue());
      updatedDist.set(updatedEntry, entry.getValue().doubleValue(),
          ((MutableDoubleCount) entry.getValue()).count);
    }

    Preconditions
        .checkState(updatedDist.getTotalCount() == this.numParticles);
    target.clear();
    target.incrementAll(updatedDist);
    Preconditions.checkState(((CountedDataDistribution) target)
        .getTotalCount() == this.numParticles);
  }

  /**
   * Expands the forward probabilities until numParticles paths are reached,
   * then returns the resulting weighed sample path.
   * 
   * @param hmm
   * @param observations
   * @param numParticles
   * @return
   */
  public static <T> TreeSet<HmmTransitionState<T>>
      expandForwardProbabilities(final HiddenMarkovModel<T> hmm,
        List<T> observations) {

    final ExposedHmm<T> eHmm = new ExposedHmm<T>(hmm);

    /*
     * Compute Baum-Welch smoothed distribution
     */
    final ArrayList<Vector> obsLikelihoodSequence =
        eHmm.computeObservationLikelihoods(observations);
    final ArrayList<WeightedValue<Vector>> forwardProbabilities =
        eHmm.computeForwardProbabilities(obsLikelihoodSequence, true);

    final List<Vector> jointProbs = Lists.newArrayList();
    for (int i = 0; i < forwardProbabilities.size() - 1; i++) {
      final WeightedValue<Vector> input = forwardProbabilities.get(i);
      final Vector prod =
          hmm.getTransitionProbability().times(input.getValue());
      jointProbs.add(prod.scale(1d / prod.norm1()));
    }
    jointProbs
        .add(Iterables.getLast(forwardProbabilities).getValue());

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
    final TreeSet<HmmTransitionState<T>> orderedDistribution =
        Sets.newTreeSet(new Comparator<HmmTransitionState<T>>() {
          @Override
          public int compare(HmmTransitionState<T> o1,
            HmmTransitionState<T> o2) {
            final int compVal =
                Double.compare(o1.getStateLogWeight(),
                    o2.getStateLogWeight());
            return compVal == 0 ? 1 : compVal;
          }
        });
    for (final ICombinatoricsVector<Integer> combination : gen) {

      HmmTransitionState<T> currentState = null;
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
              new HmmTransitionState<T>(hmm, stateAtTime, i);
        } else {
          currentState =
              new HmmTransitionState<T>(currentState, stateAtTime, i);
        }
        currentState.setStateLogWeight(logWeightOfState);
      }
      orderedDistribution.add(currentState);
    }

    return orderedDistribution;
  }

}