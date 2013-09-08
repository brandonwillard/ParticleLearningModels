package hmm.gaussian;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import hmm.HmmTransitionState;
import hmm.HmmPlFilter;

import java.util.List;
import java.util.Random;
import java.util.TreeSet;

import org.apache.log4j.Logger;

import com.google.common.collect.Iterables;

import utils.CountedDataDistribution;
import utils.ObservedValue;
import utils.WFCountedDataDistribution;

/**
 * A Particle Learning filter for a multivariate Gaussian Dirichlet Process.
 * 
 * @author bwillard
 * 
 */
public class GaussianArHmmPlFilter extends HmmPlFilter<Double> {

  public class GaussianArHmmPlUpdater extends HmmPlUpdater<Double> {

    final protected UnivariateGaussian prior;

    public GaussianArHmmPlUpdater(HiddenMarkovModel<Double> hmm, UnivariateGaussian prior, Random rng) {
      super(hmm, rng);
      this.prior = prior;
    }

    @Override
    public double computeLogLikelihood(
      HmmTransitionState<Double> particle,
      ObservedValue<Double> observation) {

      final GaussianArTransitionState transState = (GaussianArTransitionState) particle;
      final double priorPredMean = transState.getPriorPredSuffStats().getMean();
      final double priorPredVar = transState.getPriorPredSuffStats().getVariance();
      final double logCt = -Math.log(priorPredVar)/2d - 
          0.5d * Math.pow(observation.getObservedValue() - priorPredMean, 2)/
          (priorPredVar + sigma_y2);

      return logCt;
    }

    @Override
    public WFCountedDataDistribution<HmmTransitionState<Double>> baumWelchInitialization(
        List<Double> sample, int numParticles) {
      return new WFCountedDataDistribution<>(createInitialParticles(numParticles), true);
//      return super.baumWelchInitialization(sample, numParticles);
    }

    @Override
    protected <T> List<Vector> computeSmoothedJointProbs(
        HiddenMarkovModel<T> hmm, List<T> observations) {
      return super.computeSmoothedJointProbs(hmm, observations);
    }

    @Override
    public <T> TreeSet<HmmTransitionState<T>> expandForwardProbabilities(
        HiddenMarkovModel<T> hmm, List<T> observations) {
      return super.expandForwardProbabilities(hmm, observations);
    }

    @Override
    public DataDistribution<HmmTransitionState<Double>> createInitialParticles(
        int numParticles) {
      final CountedDataDistribution<HmmTransitionState<Double>> initialParticles =
          new CountedDataDistribution<>(numParticles, true);
      for (int i = 0; i < numParticles; i++) {
        final int sampledState =
            DiscreteSamplingUtil.sampleIndexFromProbabilities(
                this.rng, this.hmm.getInitialProbability());
        final HmmTransitionState<Double> particle =
            new GaussianArTransitionState(this.hmm, sampledState,
                new ObservedValue<Double>(0, null) , this.prior);

        final double logWeight = -Math.log(numParticles);
        particle.setStateLogWeight(logWeight);
        initialParticles.increment(particle, logWeight);
      }
      return initialParticles;
    }

    @Override
    public HmmTransitionState<Double> update(
      HmmTransitionState<Double> previousParameter) {
      return previousParameter.clone();
    }

  }

  final Logger log = Logger.getLogger(GaussianArHmmPlFilter.class);

  final double[] a;
  final double[] sigma2;
  final double sigma_y2;

  public GaussianArHmmPlFilter(HiddenMarkovModel<Double> hmm,
    UnivariateGaussian prior, double[] a, double[] sigma2, double sigma_y2, Random rng, boolean resampleOnly) {
    super(resampleOnly);
    this.a = a;
    this.sigma2 = sigma2;
    this.sigma_y2 = sigma_y2;
    this.setUpdater(new GaussianArHmmPlUpdater(hmm, prior, rng));
    this.setRandom(rng);
  }

  @Override
  protected HmmTransitionState<Double> propagate(
      HmmTransitionState<Double> particle, int i, ObservedValue<Double> data) {
    /*
     * Perform the filtering step
     */
    final GaussianArTransitionState prevState = (GaussianArTransitionState) particle;
    final UnivariateGaussian priorDist = prevState.getSuffStat();
    final double priorPredMean = a[i] * priorDist.getMean();
    final double priorPredCov = a[i] * a[i] * priorDist.getVariance() + sigma2[i];
    
    final double postCovariance = 1d/(1d/sigma2[i] + 1d/sigma_y2);
    final double postMean = (priorPredMean/priorPredCov 
        + data.getObservedValue()/sigma_y2)
        * postCovariance;
    final UnivariateGaussian postDist =
        new UnivariateGaussian(postMean, postCovariance);
    final HiddenMarkovModel<Double> newHmm = prevState.getHmm();

    final GaussianArTransitionState newTransState =
        new GaussianArTransitionState(prevState, newHmm,
            i, data, postDist);

    newTransState.setPriorPredSuffStats(new UnivariateGaussian(priorPredMean, priorPredCov));
    
    return newTransState;
  }

}
