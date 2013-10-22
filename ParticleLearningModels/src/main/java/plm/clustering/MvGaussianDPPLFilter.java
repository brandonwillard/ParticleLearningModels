package plm.clustering;

import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.AbstractParticleFilter;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.MultivariateStudentTDistribution;
import gov.sandia.cognition.statistics.distribution.NormalInverseWishartDistribution;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.ObjectUtil;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.statslibextensions.statistics.ExtSamplingUtils;

/**
 * A Particle Learning filter for a multivariate Gaussian Dirichlet Process.
 * 
 * @author bwillard
 * 
 */
public class MvGaussianDPPLFilter extends AbstractParticleFilter<Vector, MvGaussianDPDistribution> {

  public class MultivariateGaussianDPPLUpdater extends AbstractCloneableSerializable
      implements
        Updater<Vector, MvGaussianDPDistribution> {

    final private NormalInverseWishartDistribution centeringDistPrior;
    final private double dpAlphaPrior;
    final private Vector nCountsPrior;
    final private Random rng;
    final private List<MultivariateGaussian> priorComponents;

    public MultivariateGaussianDPPLUpdater(List<MultivariateGaussian> priorComponents,
        NormalInverseWishartDistribution centeringDistPrior, double dpAlphaPrior, Vector nCounts,
        Random rng) {
      this.priorComponents = priorComponents;
      this.centeringDistPrior = centeringDistPrior;
      this.dpAlphaPrior = dpAlphaPrior;
      this.nCountsPrior = nCounts;
      this.rng = rng;
    }

    /**
     * In the case of a Particle Learning model, such as this, the prior predictive log likelihood
     * is used.
     */
    @Override
    public double computeLogLikelihood(MvGaussianDPDistribution particle, Vector observation) {

      final double[] componentPriorPredLogLikelihoods =
          new double[particle.getDistributionCount() + 1];

      /*
       * Evaluate the log likelihood for a new component.
       */
      final NormalInverseWishartDistribution centeringDist = particle.getCenteringDistribution();
      final double newComponentPriorPredDof =
          2d * centeringDist.getInverseWishart().getDegreesOfFreedom()
              - centeringDist.getInputDimensionality() + 1d;
      final double kappa = centeringDist.getCovarianceDivisor();
      final Matrix newComponentPriorPredPrecision =
          centeringDist.getInverseWishart().getInverseScale()
              .scale(2d * (kappa + 1d) / (kappa * newComponentPriorPredDof));
      final MultivariateStudentTDistribution newComponentPriorPred =
          new MultivariateStudentTDistribution(newComponentPriorPredDof, centeringDist
              .getGaussian().getMean(), newComponentPriorPredPrecision.inverse());

      final double newComponentLogLikelihood =
          Math.log(particle.getAlpha()) - Math.log(particle.getAlpha() + particle.getIndex())
              + newComponentPriorPred.getProbabilityFunction().logEvaluate(observation);
      componentPriorPredLogLikelihoods[0] = newComponentLogLikelihood;

      double totalLogLikelihood = newComponentLogLikelihood;

      /*
       * Now, evaluate log likelihood for the current mixture components
       */
      int n = 0;
      for (final MultivariateGaussian component : particle.getDistributions()) {

        final double componentN = particle.getCounts().getElement(n);
        final double componentPriorPredDof =
            2d * centeringDist.getInverseWishart().getDegreesOfFreedom() + componentN
                - centeringDist.getInputDimensionality() + 1d;
        final Vector componentPriorPredMean =
            centeringDist.getGaussian().getMean().scale(kappa)
                .plus(component.getMean().scale(componentN)).scale(1d / (kappa + componentN));


        final Vector componentCenteringMeanDiff =
            centeringDist.getGaussian().getMean().minus(component.getMean());
        final Matrix componentD =
            component.getCovariance().plus(
                componentCenteringMeanDiff.outerProduct(componentCenteringMeanDiff).scale(
                    kappa * componentN / (kappa + componentN)));

        final Matrix componentPriorPredCovariance =
            centeringDist
                .getInverseWishart()
                .getInverseScale()
                .plus(componentD.scale(1d / 2d))
                .scale(
                    2d * (kappa + componentN + 1d) / ((kappa + componentN) * componentPriorPredDof));

        // FIXME TODO avoid this inverse!
        final MultivariateStudentTDistribution componentPriorPred =
            new MultivariateStudentTDistribution(componentPriorPredDof, componentPriorPredMean,
                componentPriorPredCovariance.inverse());

        final double componentLogLikelihood =
            Math.log(componentN) - Math.log(particle.getAlpha() + particle.getIndex())
                + componentPriorPred.getProbabilityFunction().logEvaluate(observation);

        componentPriorPredLogLikelihoods[n + 1] = componentLogLikelihood;
        totalLogLikelihood = LogMath.add(totalLogLikelihood, componentLogLikelihood);
        n++;
      }

      particle.setComponentPriorPredLogLikelihoods(componentPriorPredLogLikelihoods,
          totalLogLikelihood);

      return totalLogLikelihood;
    }

    @Override
    public DataDistribution<MvGaussianDPDistribution> createInitialParticles(int numParticles) {

      final DefaultDataDistribution<MvGaussianDPDistribution> initialParticles =
          new DefaultDataDistribution<MvGaussianDPDistribution>(numParticles);
      for (int i = 0; i < numParticles; i++) {
        final MvGaussianDPDistribution particleMvgDPDist =
            new MvGaussianDPDistribution(
                ObjectUtil.cloneSmartElementsAsArrayList(this.priorComponents),
                this.centeringDistPrior.clone(), this.dpAlphaPrior, this.nCountsPrior.clone());
        initialParticles.increment(particleMvgDPDist);
      }
      return initialParticles;
    }

    /**
     * In this model/filter, there's no need for blind samples from the predictive distribution.
     */
    @Override
    public MvGaussianDPDistribution update(MvGaussianDPDistribution previousParameter) {
      return previousParameter;
    }

  }

  public MvGaussianDPPLFilter(List<MultivariateGaussian> priorComponents,
      NormalInverseWishartDistribution centeringDistributionPrior, double dpAlphaPrior,
      Vector nCountsPrior, Random rng) {
    Preconditions.checkState(priorComponents.size() == nCountsPrior.getDimensionality());
    Preconditions.checkState(dpAlphaPrior > 0d);
    Preconditions.checkState(!centeringDistributionPrior.getInverseWishart().getInverseScale()
        .isZero());
    this.setUpdater(new MultivariateGaussianDPPLUpdater(priorComponents,
        centeringDistributionPrior, dpAlphaPrior, nCountsPrior, rng));
    this.setRandom(rng);
  }

  @Override
  public void update(DataDistribution<MvGaussianDPDistribution> target, Vector data) {
    Preconditions.checkState(target.getDomainSize() == this.numParticles);

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final double[] cumulativeLogLikelihoods = new double[this.numParticles];
    final List<MvGaussianDPDistribution> particleSupport = Lists.newArrayList(target.getDomain());
    int j = 0;
    for (final MvGaussianDPDistribution particle : particleSupport) {
      final double logLikelihood = this.updater.computeLogLikelihood(particle, data);
      cumulativeLogLikelihoods[j] =
          j > 0 ? LogMath.add(cumulativeLogLikelihoods[j - 1], logLikelihood) : logLikelihood;
      particleTotalLogLikelihood = LogMath.add(particleTotalLogLikelihood, logLikelihood);
      j++;
    }

    final List<MvGaussianDPDistribution> resampledParticles =
        ExtSamplingUtils.sampleMultipleLogScale(cumulativeLogLikelihoods, particleTotalLogLikelihood,
            particleSupport, random, this.numParticles);

    /*
     * Propagate
     */
    final DataDistribution<MvGaussianDPDistribution> updatedDist = new DefaultDataDistribution<MvGaussianDPDistribution>();
    for (final MvGaussianDPDistribution particle : resampledParticles) {

      /*
       * First, sample a mixture component index
       */
      final int componentIndex =
          ExtSamplingUtils.sampleIndexFromLogProbabilities(random,
              particle.getComponentPriorPredLogLikelihoods(),
              particle.getComponentPriorPredTotalLogLikelihood());

      final List<MultivariateGaussian> updatedComponentDists =
          (List<MultivariateGaussian>) ObjectUtil.cloneSmartElementsAsArrayList(particle
              .getDistributions());
      Vector updatedCounts;

      if (componentIndex == 0) {
        /*
         * This is the case in which we've sampled a new mixture component, so now we must create
         * it.
         */
        final MultivariateGaussian newComponentDist =
            new MultivariateGaussian(data, MatrixFactory.getDenseDefault().createMatrix(
                data.getDimensionality(), data.getDimensionality()));
        updatedComponentDists.add(newComponentDist);
        updatedCounts =
            VectorFactory.getDenseDefault().copyArray(
                Arrays.copyOf(particle.getCounts().toArray(), particle.getDistributionCount() + 1));
        updatedCounts.setElement(particle.getDistributionCount(), 1d);
      } else {
        /*
         * We've sampled an existing component, so updated the component's (sample) mean and
         * covariance.
         */
        updatedCounts = particle.getCounts().clone();
        final int adjComponentIndex = componentIndex - 1;
        final double oldComponentCount = updatedCounts.getElement(adjComponentIndex);
        final double updatedComponentCount = oldComponentCount + 1d;
        updatedCounts.setElement(adjComponentIndex, updatedComponentCount);

        final MultivariateGaussian sampledComponentDist =
            updatedComponentDists.get(adjComponentIndex);
        final Vector oldComponentMean = sampledComponentDist.getMean();
        final Vector updatedComponentMean =
            oldComponentMean.scale(oldComponentCount).plus(data).scale(1d / updatedComponentCount);
        final Matrix updatedComponentSS =
            sampledComponentDist
                .getCovariance()
                .plus(data.outerProduct(data))
                .plus(oldComponentMean.outerProduct(oldComponentMean).scale(oldComponentCount))
                .minus(
                    updatedComponentMean.outerProduct(updatedComponentMean).scale(
                        updatedComponentCount));

        sampledComponentDist.setCovariance(updatedComponentSS);
        sampledComponentDist.setMean(updatedComponentMean);
      }

      final MvGaussianDPDistribution updatedParticle =
          new MvGaussianDPDistribution(updatedComponentDists, particle.getCenteringDistribution(),
              particle.getAlpha(), updatedCounts);
      updatedParticle.setIndex(particle.getIndex() + 1);
      /*
       * Keep this data for debugging.
       */
      updatedParticle.setComponentPriorPredLogLikelihoods(
          particle.getComponentPriorPredLogLikelihoods(),
          particle.getComponentPriorPredTotalLogLikelihood());

      updatedDist.increment(updatedParticle);
    }

    target.clear();
    target.incrementAll(updatedDist);
  }
}
