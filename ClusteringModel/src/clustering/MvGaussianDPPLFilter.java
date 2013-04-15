package clustering;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
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

public class MvGaussianDPPLFilter extends 
  AbstractParticleFilter<Vector, MvGaussianDPDistribution> {

  public class MultivariateGaussianDPPLUpdater
      extends AbstractCloneableSerializable
      implements
      Updater<Vector, MvGaussianDPDistribution> {
    
    final private NormalInverseWishartDistribution centeringDistPrior;
    final private double dpAlphaPrior;
    final private Vector nCountsPrior;
    final private Random rng;
    final private List<MultivariateGaussian> priorComponents;

    public MultivariateGaussianDPPLUpdater(
      List<MultivariateGaussian> priorComponents,
      NormalInverseWishartDistribution centeringDistPrior, double dpAlphaPrior,
      Vector nCounts, Random rng) {
      this.priorComponents = priorComponents;
      this.centeringDistPrior = centeringDistPrior;
      this.dpAlphaPrior = dpAlphaPrior;
      this.nCountsPrior = nCounts;
      this.rng = rng;
    }

    /**
     * @see gov.sandia.cognition.statistics.bayesian.ParticleFilter.Updater#update(java.lang.Object)
     */
    @Override
    public MvGaussianDPDistribution update(
      MvGaussianDPDistribution previousParameter) {
      return previousParameter;
    }

    @Override
    public DataDistribution<MvGaussianDPDistribution>
        createInitialParticles(int numParticles) {
      
      DefaultDataDistribution<MvGaussianDPDistribution> initialParticles = new DefaultDataDistribution<>(numParticles);
      for (int i = 0; i < numParticles; i++) {
        MvGaussianDPDistribution particleMvgDPDist = new MvGaussianDPDistribution(
            ObjectUtil.cloneSmartElementsAsArrayList(this.priorComponents),
            this.centeringDistPrior.clone(),
            this.dpAlphaPrior, 
            this.nCountsPrior.clone());
        initialParticles.increment(particleMvgDPDist);        
      }
      return initialParticles;
    }

    /**
     * In the case of a Particle Learning model, such as this, the prior predictive log likelihood
     * is used.
     */
    @Override
    public double
        computeLogLikelihood(
          MvGaussianDPDistribution particle,
          Vector observation) {
      
      double[] componentPriorPredLogLikelihoods = new double[particle.getDistributionCount() + 1];
      
      /*
       * Evaluate the log likelihood for a new component.
       */
      NormalInverseWishartDistribution centeringDist = particle.getCenteringDistribution();
      final double newComponentPriorPredDof = 2d * centeringDist.getInverseWishart().getDegreesOfFreedom()
          - centeringDist.getInputDimensionality() + 1d;
      final double kappa = centeringDist.getCovarianceDivisor();
      Matrix newComponentPriorPredPrecision = centeringDist.getInverseWishart().getInverseScale().
          scale(2d * (kappa + 1d)/(kappa * newComponentPriorPredDof));
      MultivariateStudentTDistribution newComponentPriorPred = new MultivariateStudentTDistribution(
          newComponentPriorPredDof, centeringDist.getGaussian().getMean(), newComponentPriorPredPrecision.inverse());
      
      final double newComponentLogLikelihood = Math.log(particle.getAlpha()) - Math.log(particle.getAlpha() + particle.getIndex())
          + newComponentPriorPred.getProbabilityFunction().logEvaluate(observation);
      componentPriorPredLogLikelihoods[0] = newComponentLogLikelihood;
      
      double totalLogLikelihood = newComponentLogLikelihood;
      
      /*
       * Now, evaluate log likelihood for the current mixture components
       */
      int n = 0;
      for (MultivariateGaussian component : particle.getDistributions()) {
        
        final double componentN = particle.getCounts().getElement(n);
        final double componentPriorPredDof = 2d * centeringDist.getInverseWishart().getDegreesOfFreedom()
            + componentN - centeringDist.getInputDimensionality() + 1d;
        final Vector componentPriorPredMean = centeringDist.getGaussian().getMean().scale(kappa).
            plus(component.getMean().scale(componentN)).scale(1d/(kappa + componentN));
        
        
        Vector componentCenteringMeanDiff = centeringDist.getGaussian().getMean().minus(component.getMean());
        Matrix componentD = component.getCovariance().plus(
            componentCenteringMeanDiff.outerProduct(componentCenteringMeanDiff).scale(
                kappa * componentN / (kappa + componentN))
            );
        
        Matrix componentPriorPredCovariance = centeringDist.getInverseWishart().getInverseScale().
            plus(componentD.scale(1d/2d)).
            scale(2d * (kappa + componentN + 1d)/((kappa + componentN) * componentPriorPredDof));
        
        // FIXME TODO avoid this inverse!
        MultivariateStudentTDistribution componentPriorPred = new MultivariateStudentTDistribution(
            componentPriorPredDof, componentPriorPredMean, componentPriorPredCovariance.inverse());
        
        final double componentLogLikelihood = Math.log(componentN) 
            - Math.log(particle.getAlpha() + particle.getIndex())
            + componentPriorPred.getProbabilityFunction().logEvaluate(observation);
        
        componentPriorPredLogLikelihoods[n + 1] = componentLogLikelihood;
        totalLogLikelihood = LogMath.add(totalLogLikelihood, componentLogLikelihood);
        n++;
      }
      
      particle.setComponentPriorPredLogLikelihoods(componentPriorPredLogLikelihoods, totalLogLikelihood);
      
      return totalLogLikelihood;
    }

  }

  public MvGaussianDPPLFilter(
    List<MultivariateGaussian> priorComponents,
    NormalInverseWishartDistribution centeringDistributionPrior, 
    double dpAlphaPrior, Vector nCountsPrior, Random rng) {
    Preconditions.checkState(priorComponents.size() == nCountsPrior.getDimensionality());
    Preconditions.checkState(dpAlphaPrior > 0d);
    Preconditions.checkState(!centeringDistributionPrior.getInverseWishart().getInverseScale().isZero());
    this.setUpdater(
        new MultivariateGaussianDPPLUpdater(priorComponents, centeringDistributionPrior, dpAlphaPrior, nCountsPrior, rng));
    this.setRandom(rng);
  }

  @Override
  public
      void
      update(
        DataDistribution<MvGaussianDPDistribution> target,
        Vector data) {
    Preconditions.checkState(target.getDomainSize() == this.numParticles);
    
    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    double[] cumulativeLogLikelihoods = new double[this.numParticles]; 
    List<MvGaussianDPDistribution> particleSupport = Lists.newArrayList(target.getDomain());
    int j = 0;
    for (MvGaussianDPDistribution particle : particleSupport) {
      final double logLikelihood = this.updater.computeLogLikelihood(particle, data);
      cumulativeLogLikelihoods[j] = 
          j > 0 ? LogMath.add(cumulativeLogLikelihoods[j-1], logLikelihood) : logLikelihood;
      particleTotalLogLikelihood = LogMath.add(particleTotalLogLikelihood, logLikelihood);
      j++;               
    }
    
    List<MvGaussianDPDistribution> resampledParticles = 
        sampleMultipleLogScale(cumulativeLogLikelihoods, particleTotalLogLikelihood, particleSupport, random, 
            this.numParticles);
    
    /*
     * Propagate
     */
    DataDistribution<MvGaussianDPDistribution> updatedDist = new DefaultDataDistribution<>();
    for (MvGaussianDPDistribution particle : resampledParticles) {
      
      /*
       * First, sample a mixture component index
       */
      final int componentIndex = sampleIndexFromLogProbabilities(random, 
          particle.getComponentPriorPredLogLikelihoods(), 
          particle.getComponentPriorPredTotalLogLikelihood());
      
      List<MultivariateGaussian> updatedComponentDists = (List<MultivariateGaussian>) ObjectUtil.cloneSmartElementsAsArrayList(
          particle.getDistributions());
      Vector updatedCounts;
      
      if (componentIndex == 0) {
        /*
         * This is the case in which we've sampled a new mixture component, so now
         * we must create it.
         */
        MultivariateGaussian newComponentDist = new MultivariateGaussian(data, 
            MatrixFactory.getDenseDefault().createMatrix(data.getDimensionality(), data.getDimensionality()));
        updatedComponentDists.add(newComponentDist);
        updatedCounts = 
            VectorFactory.getDenseDefault().copyArray(
            Arrays.copyOf(particle.getCounts().toArray(), particle.getDistributionCount() + 1));
        updatedCounts.setElement(particle.getDistributionCount(), 1d);
      } else {
        /*
         * We've sampled an existing component, so updated the component's (sample) mean and covariance.
         */
        updatedCounts = particle.getCounts().clone();
        final int adjComponentIndex = componentIndex - 1;
        final double oldComponentCount = updatedCounts.getElement(adjComponentIndex);
        final double updatedComponentCount = oldComponentCount  + 1d;
        updatedCounts.setElement(adjComponentIndex, updatedComponentCount);
        
        MultivariateGaussian sampledComponentDist = updatedComponentDists.get(adjComponentIndex);
        Vector oldComponentMean = sampledComponentDist.getMean();
        Vector updatedComponentMean = oldComponentMean.scale(oldComponentCount).plus(data).scale(1d/updatedComponentCount);
        Matrix updatedComponentSS = sampledComponentDist.getCovariance().
            plus(data.outerProduct(data)).
            plus(oldComponentMean.outerProduct(oldComponentMean).scale(oldComponentCount)).
            minus(updatedComponentMean.outerProduct(updatedComponentMean).scale(updatedComponentCount));
        
        sampledComponentDist.setCovariance(updatedComponentSS);
        sampledComponentDist.setMean(updatedComponentMean);
      }
      
      MvGaussianDPDistribution updatedParticle = new MvGaussianDPDistribution(
          updatedComponentDists, particle.getCenteringDistribution(), particle.getAlpha(), 
          updatedCounts);
      updatedParticle.setIndex(particle.getIndex() + 1);
      /*
       * Keep this data for debugging.
       */
      updatedParticle.setComponentPriorPredLogLikelihoods(particle.getComponentPriorPredLogLikelihoods(),
          particle.getComponentPriorPredTotalLogLikelihood());
      
      updatedDist.increment(updatedParticle);
    }
    
    target.clear();
    target.incrementAll(updatedDist);
  }

  private
      List<MvGaussianDPDistribution>
      lowVarianceSample(
        Map<MvGaussianDPDistribution, Double> resampleParticles) {
    List<MvGaussianDPDistribution> resampledParticles = Lists.newArrayList();
    final double M = resampleParticles.size();
    final double r = random.nextDouble() / M;
    final Iterator<Entry<MvGaussianDPDistribution, Double>> pIter = resampleParticles.entrySet().iterator();
    Entry<MvGaussianDPDistribution, Double> p = pIter.next();
    double c = p.getValue();
    for (int m = 0; m < M; ++m) {
      final double U = Math.log(r + m / M);
      while (U > c && pIter.hasNext()) {
        p = pIter.next();
        c = LogMath.add(p.getValue(), c);
      }
      resampledParticles.add(p.getKey());
    }   
    return resampledParticles;
  }

  private List<MvGaussianDPDistribution> sampleMultipleLogScale(
    final double[] cumulativeLogWeights, final double logWeightSum,
    final List<MvGaussianDPDistribution> domain, final Random random,
    final int numSamples) {

    int index;
    List<MvGaussianDPDistribution> samples = Lists.newArrayListWithCapacity(numSamples);
    for (int n = 0; n < numSamples; n++) {
      double p = logWeightSum + Math.log(random.nextDouble());
      index = Arrays.binarySearch(cumulativeLogWeights, p);
      if (index < 0) {
        int insertionPoint = -index - 1;
        index = insertionPoint;
      }
      samples.add(domain.get(index));
    }
    return samples;

  }
    
  private int sampleIndexFromLogProbabilities(final Random random, final double[] logProbs, double totalLogProbs) {
    double value = Math.log(random.nextDouble());
    final int lastIndex = logProbs.length - 1;
    for (int i = 0; i < lastIndex; i++) {
      value = LogMath.subtract(value, logProbs[i] - totalLogProbs);
      if (Double.isNaN(value) || value == Double.NEGATIVE_INFINITY) {
        return i;
      }
    }
    return lastIndex;
  }
}
