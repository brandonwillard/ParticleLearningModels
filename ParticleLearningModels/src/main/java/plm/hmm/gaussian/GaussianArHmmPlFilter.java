package plm.hmm.gaussian;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;

import java.util.List;
import java.util.Random;
import java.util.TreeSet;

import org.apache.log4j.Logger;

import plm.hmm.HmmPlFilter;
import plm.hmm.HmmTransitionState;
import plm.hmm.StandardHMM;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.statslibextensions.statistics.CountedDataDistribution;
import com.statslibextensions.statistics.distribution.WFCountedDataDistribution;
import com.statslibextensions.util.ObservedValue;

/**
 * A Particle Learning filter for a multivariate Gaussian Dirichlet Process.
 * 
 * @author bwillard
 * 
 */
public class GaussianArHmmPlFilter extends HmmPlFilter<StandardHMM<Double>, GaussianArTransitionState, Double> {

  public class GaussianArHmmPlUpdater extends HmmPlUpdater<StandardHMM<Double>, GaussianArTransitionState, Double> {

    final protected UnivariateGaussian prior;

    public GaussianArHmmPlUpdater(StandardHMM<Double> hmm, UnivariateGaussian prior, Random rng) {
      super(hmm, rng);
      this.prior = prior;
    }

    @Override
    public double computeLogLikelihood(
      GaussianArTransitionState particle,
      ObservedValue<Double,Void> observation) {

      final GaussianArTransitionState transState = (GaussianArTransitionState) particle;
      final double priorPredMean = transState.getPriorPredSuffStats().getMean();
      final double priorPredVar = transState.getPriorPredSuffStats().getVariance();
      final double logCt = -Math.log(priorPredVar)/2d - 
          0.5d * Math.pow(observation.getObservedValue() - priorPredMean, 2)/
          (priorPredVar + sigma_y2);

      return logCt;
    }

    /**
     * TODO FIXME XXX: not done
     */
    @Override
    public WFCountedDataDistribution<GaussianArTransitionState> baumWelchInitialization(
        List<Double> sample, int numParticles) {
      Preconditions.checkState(false);
      return null;
    }


    public <T> List<CountedDataDistribution<HmmTransitionState<T, ?>>> computeForwardProbabilities(
        StandardHMM<T> hmm, List<T> observations) {
      
      Preconditions.checkState(false);
      List<CountedDataDistribution<HmmTransitionState<T, ?>>> results = Lists.newArrayList();
      CountedDataDistribution<HmmTransitionState<T, ?>> currentStates = 
          CountedDataDistribution.create(true);
      for(int i = 0; i < observations.size(); i++) {
        CountedDataDistribution<HmmTransitionState<T, ?>> newStates = 
            CountedDataDistribution.create(true);
        for (HmmTransitionState<T, ?> state : currentStates.getDomain()) {
          for(int j = 0; j < hmm.getNumStates(); j++) {
            HmmTransitionState<T, ?> updatedState = null;
            // TODO filter (just like the PL filter steps)
            newStates.increment(updatedState);
          }
        }
        results.add(newStates);
        currentStates = newStates;
      }
      
      return results;
    }

    @Override
    public DataDistribution<GaussianArTransitionState> createInitialParticles(
        int numParticles) {
      final CountedDataDistribution<GaussianArTransitionState> initialParticles =
          CountedDataDistribution.create(numParticles, true);
      for (int i = 0; i < numParticles; i++) {
        final int sampledState =
            DiscreteSamplingUtil.sampleIndexFromProbabilities(
                this.rng, this.priorHmm.getClassMarginalProbabilities());
        final GaussianArTransitionState particle =
            new GaussianArTransitionState(this.priorHmm, sampledState,
                ObservedValue.<Double>create(0, null) , this.prior);

        final double logWeight = -Math.log(numParticles);
        particle.setStateLogWeight(logWeight);
        initialParticles.increment(particle, logWeight);
      }
      return initialParticles;
    }

    @Override
    public GaussianArTransitionState update(
      GaussianArTransitionState priorParameter) {

      final double priorPredCov = priorParameter.getPriorPredSuffStats().getVariance();
      final double priorPredMean = priorParameter.getPriorPredSuffStats().getMean();

      final double postCovariance = 1d/(1d/priorPredCov + 1d/sigma_y2);
      final double postMean = (priorPredMean/priorPredCov 
          + priorParameter.getObservation().getObservedValue()/sigma_y2)
          * postCovariance;

      final UnivariateGaussian postDist =
          new UnivariateGaussian(postMean, postCovariance);
  
      final GaussianArTransitionState postState =
          priorParameter.clone();
      postState.setSuffStat(postDist);

      return postState;
    }

  }

  final Logger log = Logger.getLogger(GaussianArHmmPlFilter.class);

  final double[] a;
  final double[] sigma2;
  final double sigma_y2;

  public GaussianArHmmPlFilter(StandardHMM<Double> hmm,
    UnivariateGaussian prior, double[] a, double[] sigma2, double sigma_y2, Random rng, boolean resampleOnly) {
    super(resampleOnly);
    this.a = a;
    this.sigma2 = sigma2;
    this.sigma_y2 = sigma_y2;
    this.setUpdater(new GaussianArHmmPlUpdater(hmm, prior, rng));
    this.setRandom(rng);
  }

  @Override
  protected GaussianArTransitionState propagate(
      GaussianArTransitionState prevState, int predClass, ObservedValue<Double,Void> data) {
    /*
     * Perform the filtering step
     */
    final UnivariateGaussian priorDist = prevState.getSuffStat();
    final double priorPredMean = a[predClass] * priorDist.getMean();
    final double priorPredCov = a[predClass] * a[predClass] * priorDist.getVariance() + sigma2[predClass];
    
    final StandardHMM<Double> newHmm = prevState.getHmm();

    final GaussianArTransitionState newTransState =
        new GaussianArTransitionState(prevState, newHmm,
            predClass, data, null);

    newTransState.setPriorPredSuffStats(new UnivariateGaussian(priorPredMean, priorPredCov));
    
    return newTransState;
  }

}
