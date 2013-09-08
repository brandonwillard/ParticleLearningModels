package hmm.gaussian;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
import gov.sandia.cognition.statistics.bayesian.AbstractParticleFilter;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import hmm.HmmTransitionState;
import hmm.HmmTransitionState.ResampleType;

import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import utils.CountedDataDistribution;
import utils.LogMath2;
import utils.MutableDoubleCount;
import utils.ObservedValue;
import utils.SamplingUtils;
import utils.WFCountedDataDistribution;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;

/**
 * This is an implementation of Liu & Chen's MKF filter with an AR term in the
 * state evolution.
 * 
 * @author bwillar0
 *
 */
public class GaussianArHmmMkfFilter
    extends
    AbstractParticleFilter<ObservedValue<Double>, GaussianArTransitionState> {

  public static class GaussianArHmmMkfUpdater extends
      AbstractCloneableSerializable implements
      Updater<ObservedValue<Double>, GaussianArTransitionState> {

    private static final long serialVersionUID = 1675005722404209890L;

    final protected HiddenMarkovModel<Double> hmm;
    final protected Random rng;
    final protected UnivariateGaussian prior;

    public GaussianArHmmMkfUpdater(HiddenMarkovModel<Double> hmm, UnivariateGaussian prior, Random rng) {
      this.prior = prior;
      this.hmm = hmm;
      this.rng = rng;
    }

    @Override
    public double computeLogLikelihood(
      GaussianArTransitionState particle,
      ObservedValue<Double> observation) {
      return Double.NaN;
    }

    @Override
    public DataDistribution<GaussianArTransitionState>
        createInitialParticles(int numParticles) {
      final CountedDataDistribution<GaussianArTransitionState> initialParticles =
          new CountedDataDistribution<>(numParticles, true);
      for (int i = 0; i < numParticles; i++) {
        final int sampledState =
            DiscreteSamplingUtil.sampleIndexFromProbabilities(
                this.rng, this.hmm.getInitialProbability());
        final GaussianArTransitionState particle =
            new GaussianArTransitionState(this.hmm, sampledState,
                new ObservedValue<Double>(-1l, null), prior);

        final double logWeight = -Math.log(numParticles);
        particle.setStateLogWeight(logWeight);
        initialParticles.increment(particle, logWeight);
      }
      return initialParticles;
    }

    @Override
    public GaussianArTransitionState update(
      GaussianArTransitionState previousParameter) {
      return previousParameter;
    }
  }

  private static final long serialVersionUID = 2271089378484039661L;
  
  final double[] a;
  final double[] sigma2;
  final double sigma_y2;

  public GaussianArHmmMkfFilter(HiddenMarkovModel<Double> hmm,
      UnivariateGaussian prior, double[] a, double[] sigma2, 
      double sigma_y2, Random rng) {
    super();
    this.random = rng;
    this.setUpdater(new GaussianArHmmMkfUpdater(hmm, prior, rng));
    this.a = a;
    this.sigma2 = sigma2;
    this.sigma_y2 = sigma_y2;
  }

  @Override
  public void update(
    DataDistribution<GaussianArTransitionState> target,
    ObservedValue<Double> data) {

    final CountedDataDistribution<GaussianArTransitionState> propogatedParticles = new CountedDataDistribution<>(true);
    for (final GaussianArTransitionState particle : target.getDomain()) {
      final HiddenMarkovModel<Double> hmm = particle.getHmm();
//      final int particleCount =
//          ((CountedDataDistribution) target).getCount(particle);

      double particleLogLik = Double.NEGATIVE_INFINITY;

      final HiddenMarkovModel<Double> newHmm = hmm;
      final CountedDataDistribution<GaussianArTransitionState> transitionDist = new CountedDataDistribution<>(true);
      for (int i = 0; i < hmm.getNumStates(); i++) {

        /*
         * Perform the filtering step
         */
        final UnivariateGaussian priorDist = particle.getSuffStat();
        final double priorPredMean = a[i] * priorDist.getMean();
        final double priorPredCov = a[i] * a[i] * priorDist.getVariance() + sigma2[i];

        final double postCovariance = 1d/(1d/priorPredCov + 1d/sigma_y2);
        final double postMean = (priorPredMean/priorPredCov + data.getObservedValue()/sigma_y2)
            * postCovariance;

        final UnivariateGaussian postDist =
            new UnivariateGaussian(postMean, postCovariance);

        final GaussianArTransitionState transState =
            new GaussianArTransitionState(particle, newHmm,
                i, data, postDist);
        
        /*
         * Using the filtered results, produce the state likelihood 
         */
        final double logCt = -Math.log(priorPredCov)/2d - 
            0.5d * Math.pow(data.getObservedValue() - priorPredMean, 2)/
            (priorPredCov + sigma_y2);

        final double transStateLogLik =
            logCt + Math.log(hmm.getTransitionProbability().getElement(
                    i, particle.getState()));
        particleLogLik = LogMath2.add(particleLogLik, transStateLogLik);
        transitionDist.increment(transState, transStateLogLik);
      }

      // factor in the prior probability
      particleLogLik += target.getLogFraction(particle);

      final GaussianArTransitionState newState = transitionDist.sample(random);
//      final double transitionLogLik = transitionDist.getLogFraction(newState);
      newState.setStateLogWeight(particleLogLik);

      propogatedParticles.increment(newState, particleLogLik);
    }

    // TODO determine when to resample
    final boolean resample = this.computeEffectiveParticles(propogatedParticles)
        /this.numParticles < 0.80;
    final ResampleType resampleType = resample ? ResampleType.REPLACEMENT : ResampleType.NONE;
    CountedDataDistribution<GaussianArTransitionState> resampledDist;
    if (resample) {
      resampledDist = new CountedDataDistribution<>(true);
      resampledDist.incrementAll(propogatedParticles.sample(random, this.numParticles));
    } else {
      resampledDist = propogatedParticles;
    }

    target.clear();
    for (GaussianArTransitionState state : resampledDist.getDomain()) {
      final int count = resampledDist.getCount(state);
      final double logProb = resampledDist.getLogFraction(state);
      state.setResampleType(resampleType);
      ((CountedDataDistribution<GaussianArTransitionState>)target).set(
          state, logProb, count);
    }
    Preconditions.checkState(((CountedDataDistribution) target)
        .getTotalCount() == this.numParticles);
  }

}