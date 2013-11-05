package plm.regularization.lasso;

import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.AbstractParticleFilter;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.ExponentialDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;

import java.util.List;
import java.util.Random;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.statslibextensions.statistics.ExtSamplingUtils;
import com.statslibextensions.statistics.distribution.ScaledInverseGammaCovDistribution;
import com.statslibextensions.util.ObservedValue;

/**
 * A Particle Learning filter for a multivariate Gaussian Dirichlet Process.
 * 
 * @author bwillard
 * 
 */
public class LassoRegressionPLFilter
    extends AbstractParticleFilter<ObservedValue<Vector, Matrix>, LassoRegressionDistribution> {

  public class LassoRegressionPLUpdater extends AbstractCloneableSerializable
      implements
        Updater<ObservedValue<Vector, Matrix>, LassoRegressionDistribution> {

    private final Random rng;
    private final MultivariateGaussian priorBeta;
    private final ScaledInverseGammaCovDistribution priorBetaCov;
    private final ScaledInverseGammaCovDistribution augLassoDist;

    public LassoRegressionPLUpdater(Random rng, MultivariateGaussian priorBeta,
        ScaledInverseGammaCovDistribution priorBetaCov,
        ScaledInverseGammaCovDistribution augLassoDist) {
      super();
      this.rng = rng;
      this.priorBeta = priorBeta;
      this.priorBetaCov = priorBetaCov;
      this.augLassoDist = augLassoDist;
    }

    /**
     * In the case of a Particle Learning model, such as this, the prior predictive log 
     * likelihood is used.<br>
     */
    @Override
    public double computeLogLikelihood(LassoRegressionDistribution particle,
        ObservedValue<Vector, Matrix> observation) {


      final Vector y = observation.getObservedValue();
      /*
       * Now, we compute the predictive dist, so that we can evaluate the likelihood.
       */
      final Matrix X = observation.getObservedData();
      final Vector priorPredObsMean = X.times(particle.getPriorBeta().getMean());
      Matrix priorPredObsCov = X.times(
          particle.getPriorBeta().getCovariance())
          .times(X.transpose()).plus(particle.augLassoSample);
      priorPredObsCov.times(particle.priorObsCovSample);

      final MultivariateGaussian priorPredictiveObsDist=
          new MultivariateGaussian(priorPredObsMean, priorPredObsCov);

      return priorPredictiveObsDist.getProbabilityFunction().logEvaluate(y);
    }

    @Override
    public DataDistribution<LassoRegressionDistribution> createInitialParticles(int numParticles) {

      final DefaultDataDistribution<LassoRegressionDistribution> initialParticles =
          new DefaultDataDistribution<LassoRegressionDistribution>(numParticles);
      for (int i = 0; i < numParticles; i++) {
        final Matrix priorBetaCovSmpl = this.priorBetaCov.sample(this.rng);
        final Matrix augLassoSmpl = this.augLassoDist.sample(this.rng);
        final LassoRegressionDistribution particleMvgDPDist =
            new LassoRegressionDistribution(this.priorBeta.clone(), 
                this.priorBetaCov.clone(), augLassoSmpl, priorBetaCovSmpl);
        initialParticles.increment(particleMvgDPDist);
      }
      return initialParticles;
    }

    /**
     * In this model/filter, there's no need for blind samples from the predictive distribution.
     */
    @Override
    public LassoRegressionDistribution update(LassoRegressionDistribution previousParameter) {
      return previousParameter;
    }

  }

  private final ScaledInverseGammaCovDistribution priorBetaCov;
  private final ScaledInverseGammaCovDistribution augLassoDist;

  public LassoRegressionPLFilter(Random rng, MultivariateGaussian priorBeta,
      ScaledInverseGammaCovDistribution priorBetaCov,
      ScaledInverseGammaCovDistribution augLassoDist) {
    super();
    this.setUpdater(new LassoRegressionPLUpdater(rng, priorBeta, priorBetaCov, augLassoDist));
    this.setRandom(rng);
    this.priorBetaCov = priorBetaCov;
    this.augLassoDist = augLassoDist;
  }

  @Override
  public void update(DataDistribution<LassoRegressionDistribution> target,
      ObservedValue<Vector, Matrix> observation) {
    Preconditions.checkState(target.getDomainSize() == this.numParticles);

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final double[] cumulativeLogLikelihoods = new double[this.numParticles];
    final List<LassoRegressionDistribution> particleSupport =
        Lists.newArrayListWithExpectedSize(this.numParticles);
    int j = 0;
    for (final LassoRegressionDistribution particle : target.getDomain()) {
      LassoRegressionDistribution priorPredictedParticle = getPriorPredictiveParticle(particle, observation);
      final double logLikelihood = this.updater.computeLogLikelihood(priorPredictedParticle, observation);
      cumulativeLogLikelihoods[j] =
          j > 0 ? LogMath.add(cumulativeLogLikelihoods[j - 1], logLikelihood) : logLikelihood;
      particleTotalLogLikelihood = LogMath.add(particleTotalLogLikelihood, logLikelihood);
      particleSupport.add(priorPredictedParticle);
      j++;
    }

    final List<LassoRegressionDistribution> resampledParticles =
        ExtSamplingUtils.sampleMultipleLogScale(cumulativeLogLikelihoods, particleTotalLogLikelihood,
            particleSupport, random, this.numParticles);

    /*
     * Propagate
     */
    final DataDistribution<LassoRegressionDistribution> updatedDist =
        new DefaultDataDistribution<LassoRegressionDistribution>();
    for (final LassoRegressionDistribution particle : resampledParticles) {

      final MultivariateGaussian postBeta = particle.getPriorBeta().clone();

      final ScaledInverseGammaCovDistribution postObsCov = particle.getPriorObsCov();

      final Matrix augCovLassoSample = this.augLassoDist.sample(random);
      final Matrix obsCovSample = postObsCov.sample(random);
      
      /*
       * Update sufficient stats.
       * TODO FIXME not done
       */
      

      final LassoRegressionDistribution updatedParticle =
          new LassoRegressionDistribution(postBeta, postObsCov,
              augCovLassoSample, obsCovSample);

      updatedDist.increment(updatedParticle);
    }

    target.clear();
    target.incrementAll(updatedDist);
  }

  private LassoRegressionDistribution getPriorPredictiveParticle(
      LassoRegressionDistribution particle,
      ObservedValue<Vector, Matrix> observation) {
    return null;
  }

  private void updateCovariancePrior(ScaledInverseGammaCovDistribution prior, Vector meanError) {
    final double newScale = prior.getInverseGammaDist().getScale() + 1d;
    final double augObsErrorMatrix = meanError.dotProduct(meanError);
    final double newShape = prior.getInverseGammaDist().getShape() + augObsErrorMatrix;
    prior.getInverseGammaDist().setScale(newScale);
    prior.getInverseGammaDist().setShape(newShape);
  }
}
