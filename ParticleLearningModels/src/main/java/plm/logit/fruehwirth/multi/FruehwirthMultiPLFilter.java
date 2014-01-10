package plm.logit.fruehwirth.multi;

import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.AbstractParticleFilter;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;

import java.util.Comparator;
import java.util.Map.Entry;
import java.util.Random;
import java.util.TreeMap;

import plm.logit.fruehwirth.multi.FruehwirthMultiParticle;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.primitives.Doubles;
import com.statslibextensions.statistics.ExtSamplingUtils;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;
import com.statslibextensions.statistics.distribution.FruewirthSchnatterEV1Distribution;
import com.statslibextensions.statistics.distribution.WFCountedDataDistribution;
import com.statslibextensions.util.ObservedValue;

/**
 * Particle filter implementing a multinomial logistic regression with water-filling.
 * The multinomial logistic follows Fruehwirth-Schnatter.
 * 
 * @author bwillard
 * 
 */
public class FruehwirthMultiPLFilter extends AbstractParticleFilter<ObservedValue<Vector, Matrix>, FruehwirthMultiParticle> {

  public class FruehwirthMultiPLUpdater extends AbstractCloneableSerializable
      implements Updater<ObservedValue<Vector, Matrix>, FruehwirthMultiParticle> {

    final protected Random rng;
    final protected FruewirthSchnatterEV1Distribution evDistribution;
    final protected KalmanFilter initialFilter;
    final protected MultivariateGaussian initialPrior;
    final protected int numCategories;

    public FruehwirthMultiPLUpdater(
        KalmanFilter initialFilter, 
        MultivariateGaussian initialPrior, 
        FruewirthSchnatterEV1Distribution evDistribution, 
        int numCategories, Random rng) {
      this.initialPrior = initialPrior;
      this.evDistribution = evDistribution;
      this.initialFilter = initialFilter;
      this.numCategories = numCategories;
      this.rng = rng;
    }

    /**
     * 
     * @param particle
     * @param observation
     * @return
     */
    @Override
    public double computeLogLikelihood(FruehwirthMultiParticle particle, 
        ObservedValue<Vector, Matrix> observation) {
      
      /*
       * TODO when using FS aug sampling, we should probably go all the way
       * and replicate their beta sampling.
       * That would require a change here...
       */
      final MultivariateGaussian predictivePrior = particle.getLinearState().clone();
      KalmanFilter kf = particle.getRegressionFilter(particle.getCategoryId());
      final Matrix G = kf.getModel().getA();
      predictivePrior.setMean(G.times(predictivePrior.getMean()));
      predictivePrior.setCovariance(
          G.times(predictivePrior.getCovariance()).times(G.transpose())
            .plus(kf.getModelCovariance()));
      final Matrix F = kf.getModel().getC();

      final UnivariateGaussian evComponent = particle.getEVcomponent();
      final double predPriorObsMean = F.times(predictivePrior.getMean()).getElement(0)
          + evComponent.getMean();
      final double predPriorObsCov = F.times(predictivePrior.getCovariance()).times(F.transpose())
          .plus(kf.getMeasurementCovariance()).getElement(0, 0);
      particle.setPriorPredMean(predPriorObsMean);
      particle.setPriorPredCov(predPriorObsCov);

      double logLikelihood = UnivariateGaussian.PDF.logEvaluate(
            particle.getAugResponseSample().getElement(0), 
            predPriorObsMean, predPriorObsCov);
      
      return logLikelihood;
    }

    @Override
    public DataDistribution<FruehwirthMultiParticle> createInitialParticles(int numParticles) {

      final DataDistribution<FruehwirthMultiParticle> initialParticles =
          CountedDataDistribution.create(true);
      for (int i = 0; i < numParticles; i++) {
        
        final MultivariateGaussian initialPriorState = initialPrior.clone();
        final KalmanFilter kf = this.initialFilter.clone();
        
        /*
         * Without an observation, start with any category...
         */
        final int categoryId = this.rng.nextInt(this.numCategories);
        final FruehwirthMultiParticle particleMvgDPDist =
            new FruehwirthMultiParticle(null, 
                kf, initialPriorState, null,
                categoryId);

        initialParticles.increment(particleMvgDPDist);
      }
      return initialParticles;
    }

    /**
     * In this model/filter, there's no need for blind samples from the predictive distribution.
     */
    @Override
    public FruehwirthMultiParticle update(FruehwirthMultiParticle previousParameter) {
      return previousParameter;
    }

  }

  final protected FruewirthSchnatterEV1Distribution evDistribution = 
      new FruewirthSchnatterEV1Distribution();
  final protected KalmanFilter initialFilter;
  final protected int numCategories;
  
  /**
   * Estimate a dynamic multinomial logit model using water-filling over a
   * EV(1) mixture approximation (i.e. Fruehwirth-Schnatter (2007)).
   * 
   * @param initialPrior
   * @param F
   * @param G
   * @param modelCovariance
   * @param rng
   * @param fsResponseSampling determines if Fruehwirth-Schnatter's method of augmented response
   * sampling should be used, or Rao-Blackwellization and sampling.
   */
  public FruehwirthMultiPLFilter(
      MultivariateGaussian initialPrior,
      Matrix F, Matrix G, Matrix  modelCovariance, 
      int numCategories, Random rng) {
    Preconditions.checkArgument(F.getNumRows() == 1);
    Preconditions.checkArgument(F.getNumColumns() == G.getNumRows());
    Preconditions.checkArgument(G.getNumColumns() == modelCovariance.getNumRows());
    this.initialFilter = new KalmanFilter(
            new LinearDynamicalSystem(
                G,
                MatrixFactory.getDefault().createMatrix(G.getNumRows(), G.getNumColumns()),
                F),
            modelCovariance,
            MatrixFactory.getDefault().copyArray(new double[][] {{0}})    
          );
    this.numCategories = numCategories;
    this.setUpdater(new FruehwirthMultiPLUpdater(initialFilter, initialPrior,
        evDistribution, numCategories, rng));
    this.setRandom(rng);
  }

  @Override
  public void update(DataDistribution<FruehwirthMultiParticle> target, ObservedValue<Vector, Matrix> data) {
    Preconditions.checkState(target.getDomainSize() == this.numParticles);

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final double prevTotalLogLikelihood = target.getTotal();
    /*
     * TODO: I like the idea of this tree-map, but not the
     * equality killing comparator.  This probably kills
     * the efficiency of the hashing, but i suppose we don't 
     * actually use that here!
     */
    TreeMap<Double, FruehwirthMultiParticle> particleTree = Maps.
        <Double, Double, FruehwirthMultiParticle>newTreeMap(
        new Comparator<Double>() {
          @Override
          public int compare(Double o1, Double o2) {
            return o1 < o2 ? 1 : -1;
          }
        });
    for (Entry<FruehwirthMultiParticle, ? extends Number> particleEntry : target.asMap().entrySet()) {
      final FruehwirthMultiParticle particle = particleEntry.getKey();

      /*
       * Fruewirth-Schnatter's method for sampling...
       * TODO do we sample new y's for each particle?
       * TODO when using FS aug sampling, we should probably go all the way
       * and replicate their beta sampling.
       * That would require a change here...
       */
      final double U = this.random.nextDouble();
      double lambdaSum = 0d;
      Vector sampledAugResponse = VectorFactory.getDefault().createVector(this.numCategories);
      for (int i = 0; i < this.numCategories; i++) {
  //        final Vector betaSample = particle.getLinearState().sample(this.random);
        final MultivariateGaussian predictivePrior = particle.getLinearState().clone();
        KalmanFilter kf = particle.getRegressionFilter(i);
        final Matrix G = kf.getModel().getA();
        predictivePrior.setMean(G.times(predictivePrior.getMean()));
        predictivePrior.setCovariance(
            G.times(predictivePrior.getCovariance()).times(G.transpose())
              .plus(kf.getModelCovariance()));
  
        // X * beta
        final double lambda = Math.exp(data.getObservedData().times(
            predictivePrior.getMean()).getElement(0));
        lambdaSum += lambda;
        final double dSampledAugResponse = -Math.log(
            -Math.log(U)/(1d+lambdaSum)
            - (data.getObservedValue().getElement(0) > 0d 
                ? 0d : Math.log(this.random.nextDouble())/lambda));
        sampledAugResponse.setElement(i, dSampledAugResponse);
      }

      /*
       * Expand particle set over mixture components
       */
      for (int j = 0; j < 10; j++) {
        double categoriesTotalLogLikelihood = 0d;

        final UnivariateGaussian componentDist = 
            this.evDistribution.getDistributions().get(j);
  
        for (int k = 0; k < this.numCategories; k++) {
          /*
           * TODO could avoid cloning if we didn't change the measurement covariance,
           * but instead used the componentDist explicitly.
           */
          final FruehwirthMultiParticle predictiveParticle = particle.clone();
          predictiveParticle.setAugResponseSample(sampledAugResponse); 
  
          predictiveParticle.setEVcomponent(componentDist);
          
          /*
           * Update the observed data for the regression component.
           */
          predictiveParticle.getRegressionFilter(k).getModel().setC(data.getObservedData());
  
          final Matrix compVar = MatrixFactory.getDefault().copyArray(
              new double[][] {{componentDist.getVariance()}});
          predictiveParticle.getRegressionFilter(k).setMeasurementCovariance(compVar);
          
          final double logLikelihood = this.updater.computeLogLikelihood(predictiveParticle, data)
              + Math.log(this.evDistribution.getPriorWeights()[j])
              + (particleEntry.getValue().doubleValue() - prevTotalLogLikelihood);
  
          categoriesTotalLogLikelihood = LogMath.subtract(categoriesTotalLogLikelihood, logLikelihood);
          particleTotalLogLikelihood = LogMath.add(particleTotalLogLikelihood, logLikelihood);

          particleTree.put(logLikelihood, predictiveParticle);
        }
      }
    }

//    final WFCountedDataDistribution<FruehwirthMultiParticle> resampledParticles =
//        ExtSamplingUtils.waterFillingResample(logLikelihoods, particleTotalLogLikelihood,
//            particleSupport, random, this.numParticles);

    final WFCountedDataDistribution<FruehwirthMultiParticle> resampledParticles =
        ExtSamplingUtils.waterFillingResample(
            Doubles.toArray(particleTree.keySet()), 
            particleTotalLogLikelihood,
            Lists.newArrayList(particleTree.values()), 
            random, this.numParticles);

    /*
     * Propagate
     */
    target.clear();
    for (final Entry<FruehwirthMultiParticle, MutableDouble> particleEntry : resampledParticles.asMap().entrySet()) {
      final FruehwirthMultiParticle updatedParticle = sufficientStatUpdate(particleEntry.getKey(), data);
      target.set(updatedParticle, particleEntry.getValue().value);
    }
    Preconditions.checkState(target.getDomainSize() == this.numParticles);

  }

  private FruehwirthMultiParticle sufficientStatUpdate(
      FruehwirthMultiParticle priorParticle, ObservedValue<Vector, Matrix> data) {
    final FruehwirthMultiParticle updatedParticle = priorParticle.clone();
    
    final Vector sampledAugResponse = priorParticle.getAugResponseSample();
    final KalmanFilter filter = updatedParticle.getRegressionFilter(
        priorParticle.getCategoryId()); 
    final UnivariateGaussian evComponent = updatedParticle.EVcomponent;
    // TODO we should've already set this, so it might be redundant.
    filter.setMeasurementCovariance(
        MatrixFactory.getDefault().copyArray(new double[][] {{
          evComponent.getVariance()}}));

    final MultivariateGaussian posteriorState = updatedParticle.getLinearState();
    filter.update(posteriorState, sampledAugResponse);
    
    return updatedParticle;
  }


  
  
}
