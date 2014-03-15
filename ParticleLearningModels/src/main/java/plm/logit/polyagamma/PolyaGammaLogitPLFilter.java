package plm.logit.polyagamma;

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
import com.statslibextensions.statistics.distribution.ScaledInverseGammaCovDistribution;
import com.statslibextensions.util.ExtSamplingUtils;
import com.statslibextensions.util.ObservedValue;

/**
 * 
 * @author bwillard
 * 
 */
public class PolyaGammaLogitPLFilter
    extends AbstractParticleFilter<ObservedValue<Vector, Matrix>, PolyaGammaLogitDistribution> {

  public class PolyaGammaPLUpdater extends AbstractCloneableSerializable
      implements
        Updater<ObservedValue<Vector, Matrix>, PolyaGammaLogitDistribution> {

    private final Random rng;
    private final MultivariateGaussian priorBeta;
    private final ScaledInverseGammaCovDistribution priorBetaCov;

    public PolyaGammaPLUpdater(Random rng, MultivariateGaussian priorBeta,
        ScaledInverseGammaCovDistribution priorBetaCov) {
      super();
      this.rng = rng;
      this.priorBeta = priorBeta;
      this.priorBetaCov = priorBetaCov;
    }

    /**
     * In the case of a Particle Learning model, such as this, the prior predictive log likelihood
     * is used.
     */
    @Override
    public double computeLogLikelihood(PolyaGammaLogitDistribution particle,
        ObservedValue<Vector, Matrix> observation) {


      /*
       * First, we create a Gaussian distribution out of the augmented response Note: we're only
       * working in 1D right now (a single binary response), however, we'll want to move to multiple
       * binary response next. that's why matrices and vectors are being used here.
       */
      final Vector y = observation.getObservedValue();
      final Vector invOmegaSamples = VectorFactory.getDenseDefault().createVector(1);
      final Vector z = VectorFactory.getDenseDefault().createVector(1);
      /*
       * Now, we compute the predictive log odds, so that we can evaluate the likelihood.
       */
      final Matrix x = observation.getObservedData();
      final Vector phi = x.times(particle.getPriorBeta().getMean());
      for (int i = 0; i < y.getDimensionality(); i++) {
        final double omega = PolyaGammaLogitDistribution.sample(
          phi.getElement(i), rng);
        invOmegaSamples.setElement(i, 1d / omega);
        z.setElement(i, (y.getElement(i) - 0.5d) / omega);
      }
      final Matrix zCov = MatrixFactory.getDenseDefault().createDiagonal(invOmegaSamples);

      final MultivariateGaussian augmentedPriorPredictiveLikelihood =
          new MultivariateGaussian(z, zCov);

      particle.setAugmentedResponseDistribution(augmentedPriorPredictiveLikelihood);


      particle.setPriorPredictiveMean(phi);

      return augmentedPriorPredictiveLikelihood.getProbabilityFunction().logEvaluate(phi);
    }

    @Override
    public DataDistribution<PolyaGammaLogitDistribution> createInitialParticles(int numParticles) {

      final DefaultDataDistribution<PolyaGammaLogitDistribution> initialParticles =
          new DefaultDataDistribution<PolyaGammaLogitDistribution>(numParticles);
      for (int i = 0; i < numParticles; i++) {
        final PolyaGammaLogitDistribution particleMvgDPDist =
            new PolyaGammaLogitDistribution(this.priorBeta.clone(), this.priorBetaCov.clone());
        initialParticles.increment(particleMvgDPDist);
      }
      return initialParticles;
    }

    /**
     * In this model/filter, there's no need for blind samples from the predictive distribution.
     */
    @Override
    public PolyaGammaLogitDistribution update(PolyaGammaLogitDistribution previousParameter) {
      return previousParameter;
    }

  }

  public PolyaGammaLogitPLFilter(Random rng, MultivariateGaussian priorBeta,
      ScaledInverseGammaCovDistribution priorBetaCov) {
    super();
    this.setUpdater(new PolyaGammaPLUpdater(rng, priorBeta, priorBetaCov));
    this.setRandom(rng);
  }

  @Override
  public void update(DataDistribution<PolyaGammaLogitDistribution> target,
      ObservedValue<Vector, Matrix> observation) {
    Preconditions.checkState(target.getDomainSize() == this.numParticles);

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final double[] cumulativeLogLikelihoods = new double[this.numParticles];
    final List<PolyaGammaLogitDistribution> particleSupport =
        Lists.newArrayList(target.getDomain());
    int j = 0;
    for (final PolyaGammaLogitDistribution particle : particleSupport) {
      final double logLikelihood = this.updater.computeLogLikelihood(particle, observation);
      cumulativeLogLikelihoods[j] =
          j > 0 ? LogMath.add(cumulativeLogLikelihoods[j - 1], logLikelihood) : logLikelihood;
      particleTotalLogLikelihood = LogMath.add(particleTotalLogLikelihood, logLikelihood);
      j++;
    }

    final List<PolyaGammaLogitDistribution> resampledParticles =
        ExtSamplingUtils.sampleReplaceCumulativeLogScale(cumulativeLogLikelihoods, 
            particleSupport, random, this.numParticles);

    /*
     * Propagate
     */
    final DataDistribution<PolyaGammaLogitDistribution> updatedDist =
        new DefaultDataDistribution<PolyaGammaLogitDistribution>();
    for (final PolyaGammaLogitDistribution particle : resampledParticles) {
      final MultivariateGaussian augResponseDist = particle.getAugmentedResponseDistribution();

      final MultivariateGaussian updatedBetaMean = particle.getPriorBeta().clone();

      final List<Double> lambdaSamples =
          new ExponentialDistribution(2).sample(random, updatedBetaMean.getInputDimensionality());
      final Matrix lambdaSamplesMatrix =
          MatrixFactory.getDenseDefault().createDiagonal(
              VectorFactory.getDenseDefault().copyValues(lambdaSamples));
      
      /*
       * To perform the following updates, we need smoothed joint samples of the previous and
       * current states (beta and the global mean), i.e. a draw from (x_t, x_{t-1} | y_t). FIXME XXX
       * The above isn't currently being done
       */
      final MultivariateGaussian priorBetaSmoothedDist = getSmoothedPriorDist(particle.getPriorBeta(),
        augResponseDist, observation, particle.getPriorPredictiveMean());
      final Vector priorBetaSmoothedSample = priorBetaSmoothedDist.sample(random);
      final MultivariateGaussian postBetaSmoothedDist = getSmoothedPostDist(particle.getPriorBeta(),
        augResponseDist, observation, particle.getPriorPredictiveMean());
      final Vector postBetaSmoothedSample = postBetaSmoothedDist.sample(random);

      final Vector priorGlobalMeanSample = particle.getPriorBeta().getMean();
      final Vector postGlobalMeanSample = particle.getPriorBeta().sample(random);

      /*
       * Perform the actual Gaussian Bayes update. FIXME This is a very poor implementation.
       */
      mvGaussianBayesUpdate(augResponseDist, priorGlobalMeanSample,
          updatedBetaMean, observation.getObservedData());

      final Vector betaMeanError = postBetaSmoothedSample.minus(priorBetaSmoothedSample);
      final ScaledInverseGammaCovDistribution updatedBetaCov = particle.getPriorBetaCov().clone();
      updateCovariancePrior(updatedBetaCov, betaMeanError);
      final Matrix betaCovSmpl = updatedBetaCov.sample(random);
      Preconditions.checkState(betaCovSmpl.getElement(0, 0) >= 0d);
      updatedBetaMean.setCovariance(lambdaSamplesMatrix.times(betaCovSmpl
          .times(updatedBetaMean.getCovariance())));

      /*
       * Now, do the above for the the global mean term.
       */
      final MultivariateGaussian updatedGlobalMean =
          particle.getPriorBeta().times(particle.getAugmentedResponseDistribution());

      mvGaussianBayesUpdate(augResponseDist,
          observation.getObservedData().times(priorBetaSmoothedSample), updatedGlobalMean,
          MatrixFactory.getDenseDefault().createIdentity(
            augResponseDist.getInputDimensionality(), augResponseDist.getInputDimensionality()));

      final Vector globalMeanError = postGlobalMeanSample.minus(priorGlobalMeanSample);
      final ScaledInverseGammaCovDistribution updatedGlobalMeanCov =
          particle.getPriorBetaCov().clone();
      updateCovariancePrior(updatedGlobalMeanCov, globalMeanError);
      final Matrix globalMeanCovSmpl = updatedGlobalMeanCov.sample(random)
          .times(updatedGlobalMean.getCovariance());
      Preconditions.checkState(globalMeanCovSmpl.getElement(0, 0) > 0d);
      updatedGlobalMean.setCovariance(globalMeanCovSmpl);

      final PolyaGammaLogitDistribution updatedParticle =
          new PolyaGammaLogitDistribution(updatedGlobalMean, updatedGlobalMeanCov);

      updatedDist.increment(updatedParticle);
    }

    target.clear();
    target.incrementAll(updatedDist);
  }

  private MultivariateGaussian getSmoothedPostDist(MultivariateGaussian postBeta, 
                                                   MultivariateGaussian augResponseDist, 
                                                   ObservedValue<Vector, Matrix> observation, 
                                                   Vector obsMeanAdj) {
    final Matrix C = postBeta.getCovariance();
    final Vector m = postBeta.getMean();
    
    // System design
    final Matrix F = observation.getObservedData();
    final Matrix G = MatrixFactory.getDefault().createIdentity(m.getDimensionality(), m.getDimensionality());
    final Matrix Omega = MatrixFactory.getDefault().createIdentity(m.getDimensionality(), m.getDimensionality()); 
    
    // Observation suff. stats 
    final Matrix Sigma = augResponseDist.getCovariance();
    final Vector y = augResponseDist.getMean().minus(obsMeanAdj);
    
    final Vector a = G.times(m);
    final Matrix R = Omega;
    
    final Matrix W = F.times(Omega).times(F.transpose()).plus(Sigma);
    final Matrix FG = F.times(G);
    final Matrix A = FG.times(R).times(FG.transpose()).plus(W);
    final Matrix Wtil =
        A.transpose().solve(FG.times(R.transpose())).transpose();

    final Vector aSmooth = a.plus(Wtil.times(y.minus(FG.times(a))));
    final Matrix RSmooth =
        R.minus(Wtil.times(A).times(Wtil.transpose()));
    
    return new MultivariateGaussian(aSmooth, RSmooth);
  }

  private MultivariateGaussian getSmoothedPriorDist(MultivariateGaussian priorBeta, 
                                                    MultivariateGaussian augResponseDist, 
                                                    ObservedValue<Vector, Matrix> observation, Vector obsMeanAdj) {
    // Prior suff. stats 
    final Matrix C = priorBeta.getCovariance();
    final Vector m = priorBeta.getMean();
    
    // System design
    final Matrix F = observation.getObservedData();
    final Matrix G = MatrixFactory.getDefault().createIdentity(m.getDimensionality(), m.getDimensionality());
    final Matrix Omega = MatrixFactory.getDefault().createIdentity(m.getDimensionality(), m.getDimensionality()); 
    
    // Observation suff. stats 
    final Matrix Sigma = augResponseDist.getCovariance();
    final Vector y = augResponseDist.getMean().minus(obsMeanAdj);
    
    final Matrix W = F.times(Omega).times(F.transpose()).plus(Sigma);
    final Matrix FG = F.times(G);
    final Matrix A = FG.times(C).times(FG.transpose()).plus(W);
    final Matrix Wtil =
        A.transpose().solve(FG.times(C.transpose())).transpose();

    final Vector mSmooth = m.plus(Wtil.times(y.minus(FG.times(m))));
    final Matrix CSmooth =
        C.minus(Wtil.times(A).times(Wtil.transpose()));
    return new MultivariateGaussian(mSmooth, CSmooth);
  }

  /**
   * Copied from MultivariateGaussian.times(MultivariateGaussian). TODO: All these inverses are bad.
   * Use/track Cholesky decomps, or SVD.
   * 
   * @param obsDist
   * @param prior
   * @param design
   * @return
   */
  private void mvGaussianBayesUpdate(MultivariateGaussian obsDist,
      Vector obsMeanComponentAdjustment, MultivariateGaussian prior, Matrix X) {
    final Vector m1 = prior.getMean();
    final Matrix c1inv = prior.getCovarianceInverse();

    final Vector m2 = obsDist.getMean().plus(obsMeanComponentAdjustment);
    final Matrix Xt = X.transpose();
    final Matrix c2inv = Xt.times(obsDist.getCovarianceInverse()).times(X);

    final Matrix Cinv = c1inv.plus(c2inv);
    final Matrix C = Cinv.inverse();

    final Vector m = C.times(c1inv.times(m1).plus(
      Xt.times(obsDist.getCovarianceInverse()).times(m2)));

    prior.setMean(m);
    prior.setCovariance(C);
  }

  private void updateCovariancePrior(ScaledInverseGammaCovDistribution prior, Vector meanError) {
    final double newScale = prior.getInverseGammaDist().getScale() + 1d;
    final double augObsErrorMatrix = meanError.dotProduct(meanError);
    final double newShape = prior.getInverseGammaDist().getShape() + augObsErrorMatrix;
    prior.getInverseGammaDist().setScale(newScale);
    prior.getInverseGammaDist().setShape(newShape);
  }
}
