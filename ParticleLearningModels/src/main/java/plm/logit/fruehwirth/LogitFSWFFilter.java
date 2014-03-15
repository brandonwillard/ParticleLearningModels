package plm.logit.fruehwirth;

import gov.sandia.cognition.math.LogMath;
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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;
import com.statslibextensions.statistics.distribution.FruewirthSchnatterEV1Distribution;
import com.statslibextensions.util.ExtSamplingUtils;
import com.statslibextensions.util.ExtStatisticsUtils;
import com.statslibextensions.util.ObservedValue;

/**
 * Particle filter implementing a logistic regression with water-filling.
 * The logistic mixture approximation follows Fruehwirth-Schnatter's EV mixture.
// * The estimation scheme is mostly the same, except when sampling the upper utility 
// * values, we use the prior predictive mean of beta instead of a sample.
 * 
 * @author bwillard
 * 
 */
public class LogitFSWFFilter extends AbstractParticleFilter<ObservedValue<Vector, Matrix>, LogitMixParticle> {

  public class LogitFSWFUpdater extends AbstractCloneableSerializable
      implements
        Updater<ObservedValue<Vector, Matrix>, LogitMixParticle> {

    final protected Random rng;
    final protected FruewirthSchnatterEV1Distribution evDistribution;
    final protected KalmanFilter initialFilter;
    final protected MultivariateGaussian initialPrior;

    public LogitFSWFUpdater(
        KalmanFilter initialFilter, 
        MultivariateGaussian initialPrior, 
        FruewirthSchnatterEV1Distribution evDistribution, 
        Random rng) {
      this.initialPrior = initialPrior;
      this.evDistribution = evDistribution;
      this.initialFilter = initialFilter;
      this.rng = rng;
    }

    /**
     * 
     * Compute the observation log likelihood, p(y_t | lambda_t, beta_t)
     * 
     * @param particle
     * @param observation
     * @return
     */
    @Override
    public double computeLogLikelihood(LogitMixParticle particle, 
        ObservedValue<Vector, Matrix> observation) {
      
      // The prior pred. dist should already be conditional on the current component...
      final double upperMean = particle.getPriorPredMean();
      final double upperVar = particle.getPriorPredCov();
      double logLikelihood = Double.NEGATIVE_INFINITY; 
      for (int i = 0; i < 10; i++) {

        final UnivariateGaussian partComponent = 
            this.evDistribution.getDistributions().get(i);

        // TODO FIXME: do this in log scale...
        double partLogLik = 
            ExtStatisticsUtils.normalCdf(0d, 
            upperMean - partComponent.getMean(), 
            Math.sqrt(upperVar + partComponent.getVariance()), 
            true);
        if (!observation.getObservedValue().isZero()) {
          partLogLik = LogMath.subtract(0d, partLogLik);
        }

        // Now, tack on i's component weight
        partLogLik += Math.log(this.evDistribution.getPriorWeights()[i]);  

        logLikelihood = LogMath.add(logLikelihood, partLogLik);
      }
      
      return logLikelihood;
    }

    @Override
    public DataDistribution<LogitMixParticle> createInitialParticles(int numParticles) {

      final DataDistribution<LogitMixParticle> initialParticles =
          CountedDataDistribution.create(true);
      for (int i = 0; i < numParticles; i++) {
        
        final MultivariateGaussian initialPriorState = initialPrior.clone();
        final KalmanFilter kf = this.initialFilter.clone();
        final int componentId = this.rng.nextInt(10);
        final UnivariateGaussian evDist = this.evDistribution.
            getDistributions().get(componentId);
        
        final LogitMixParticle particleMvgDPDist =
            new LogitMixParticle(null, 
                kf, initialPriorState, evDist);
        initialParticles.increment(particleMvgDPDist);
      }
      return initialParticles;
    }

    /**
     * In this model/filter, there's no need for blind samples from the predictive distribution.
     */
    @Override
    public LogitMixParticle update(LogitMixParticle previousParameter) {
      return previousParameter;
    }

  }

  final protected FruewirthSchnatterEV1Distribution evDistribution = 
      new FruewirthSchnatterEV1Distribution();
  final protected KalmanFilter initialFilter;
  
  /**
   * Estimate a dynamic logit model using water-filling over a
   * EV(1) mixture approximation (i.e. Fruehwirth-Schnatter (2007)).
   * 
   * @param initialPrior
   * @param F
   * @param G
   * @param modelCovariance
   * @param rng
   */
  public LogitFSWFFilter(
      MultivariateGaussian initialPrior,
      Matrix F, Matrix G, Matrix  modelCovariance, 
      Random rng) {
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
    this.setUpdater(new LogitFSWFUpdater(initialFilter, initialPrior,
        evDistribution, rng));
    this.setRandom(rng);
  }

  @Override
  public void update(DataDistribution<LogitMixParticle> target, ObservedValue<Vector, Matrix> data) {

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final boolean isOne = !data.getObservedValue().isZero();

    /*
     * XXX: This treemap cannot be used for anything other than
     * what it's currently being used for, since the interface
     * contract is explicitly broken with our comparator.
     */
    TreeMap<Double, LogitMixParticle> particleTree = Maps.
        <Double, Double, LogitMixParticle>newTreeMap(
        new Comparator<Double>() {
          @Override
          public int compare(Double o1, Double o2) {
            return o1 < o2 ? 1 : -1;
          }
        });
    for (Entry<LogitMixParticle, ? extends Number> particleEntry : target.asMap().entrySet()) {
      final LogitMixParticle particle = particleEntry.getKey();

      /*
       * Fruewirth-Schnatter's method for upper utility sampling, where
       * instead of sampling the predictors, we use the mean.
       */
      final MultivariateGaussian predictivePrior = particle.getLinearState().clone();
      KalmanFilter kf = particle.getRegressionFilter();
      final Matrix G = kf.getModel().getA();
      final Matrix F = data.getObservedData();
      predictivePrior.setMean(G.times(predictivePrior.getMean()));
      predictivePrior.setCovariance(
          G.times(predictivePrior.getCovariance()).times(G.transpose())
            .plus(kf.getModelCovariance()));
      final Vector betaSample = 
          predictivePrior.sample(getRandom());
//          predictivePrior.getMean();
      final double predPriorObsMean = 
            F.times(betaSample).getElement(0);

      final int particleCount;
      if (particleEntry.getValue() instanceof MutableDoubleCount) {
        particleCount = ((MutableDoubleCount)particleEntry.getValue()).count; 
      } else {
        particleCount = 1;
      }
      for (int p = 0; p < particleCount; p++) {
        final double dSampledAugResponse = sampleAugResponse(predPriorObsMean, isOne);

        Vector sampledAugResponse = VectorFactory.getDefault().copyValues(dSampledAugResponse);

        for (int j = 0; j < 10; j++) {
          final LogitMixParticle predictiveParticle = particle.clone();
          predictiveParticle.setPreviousParticle(particle);
          predictiveParticle.setBetaSample(betaSample);
          predictiveParticle.setAugResponseSample(sampledAugResponse); 
          predictiveParticle.setLinearState(predictivePrior);

          final UnivariateGaussian componentDist = 
              this.evDistribution.getDistributions().get(j);

          predictiveParticle.setEVcomponent(componentDist);
          
          /*
           * Update the observed data for the regression component.
           */
          predictiveParticle.getRegressionFilter().getModel().setC(F);

          // TODO would be great to have a 1x1 matrix class here...
          final Matrix compVar = MatrixFactory.getDefault().copyArray(
              new double[][] {{componentDist.getVariance()}});
          predictiveParticle.getRegressionFilter().setMeasurementCovariance(compVar);
          
          final double compPredPriorObsMean = 
               F.times(betaSample).getElement(0) 
               + componentDist.getMean();
          final double compPredPriorObsCov = 
               F.times(predictivePrior.getCovariance()).times(F.transpose()).getElement(0, 0) 
               + componentDist.getVariance();
          predictiveParticle.setPriorPredMean(compPredPriorObsMean);
          predictiveParticle.setPriorPredCov(compPredPriorObsCov);

          final double logLikelihood = 
              this.updater.computeLogLikelihood(predictiveParticle, data);

//          final double compLogLikelihood = UnivariateGaussian.PDF.logEvaluate(
//                dSampledAugResponse, 
//                compPredPriorObsMean,
//                compPredPriorObsCov);

          // FIXME we're just assuming equivalent particles had equal weight
          final double priorLogWeight = particleEntry.getValue().doubleValue() 
              - Math.log(particleCount);
          
          final double jointLogLikelihood = 
              logLikelihood
              // add the weight for this component
              + Math.log(this.evDistribution.getPriorWeights()[j])
              + priorLogWeight;

          predictiveParticle.setWeight(jointLogLikelihood);

          particleTotalLogLikelihood = LogMath.add(particleTotalLogLikelihood, jointLogLikelihood);
          particleTree.put(jointLogLikelihood, predictiveParticle);
        }
      }
    }

    final CountedDataDistribution<LogitMixParticle> resampledParticles =
        ExtSamplingUtils.waterFillingResample( 
            Doubles.toArray(particleTree.keySet()), 
            particleTotalLogLikelihood,
            Lists.newArrayList(particleTree.values()), 
            random, this.numParticles);

    /*
     * Propagate
     */
    target.clear();
    for (final Entry<LogitMixParticle, ? extends Number> particleEntry : resampledParticles.asMap().entrySet()) {
      final LogitMixParticle updatedParticle = sufficientStatUpdate(
          particleEntry.getKey(), data);
      final Number value = particleEntry.getValue();
      if (particleEntry.getValue() instanceof MutableDoubleCount) {
        ((CountedDataDistribution)target).set(updatedParticle, value.doubleValue(), 
            ((MutableDoubleCount)particleEntry.getValue()).count); 
      } else {
        target.set(updatedParticle, value.doubleValue());
      }
    }
    if (target instanceof CountedDataDistribution) {
      Preconditions.checkState(((CountedDataDistribution)target).getTotalCount() == this.numParticles);
    } else {
      Preconditions.checkState(target.getDomainSize() == this.numParticles);
    }

  }

  private double sampleAugResponse(double predPriorObsMean, boolean isOne) {
    /*
     * compute X * beta
     * TODO FIXME: We really should scale the
     * predictor matrix, or something similar, otherwise,
     * the following value can be infinity.
     */
    final double lambda = Math.exp(predPriorObsMean);
    final double dSampledAugResponse;
    if (Doubles.isFinite(lambda)) {
      final double mlU = -Math.log(this.random.nextDouble());
      dSampledAugResponse = -Math.log(mlU/(1d+lambda) 
          - (!isOne ? Math.log(this.random.nextDouble())/lambda : 0d));
    } else {
      final double mlU = -Math.log(this.random.nextDouble());
      dSampledAugResponse = -Math.log(mlU/(1d+lambda) 
          - (!isOne ? Math.log(this.random.nextDouble())/lambda : 0d));
      
    }
      
    return dSampledAugResponse;
  }

  private LogitMixParticle sufficientStatUpdate(
      LogitMixParticle priorParticle, ObservedValue<Vector, Matrix> data) {
    final LogitMixParticle updatedParticle = priorParticle.clone();
    
    final KalmanFilter filter = updatedParticle.getRegressionFilter(); 
    final UnivariateGaussian evComponent = updatedParticle.EVcomponent;
    // TODO we should've already set this, so it might be redundant.
    filter.setMeasurementCovariance(
        MatrixFactory.getDefault().copyArray(new double[][] {{
          evComponent.getVariance()}}));

    final Vector sampledAugResponse = priorParticle.getAugResponseSample();
    final Vector diffAugResponse = 
        sampledAugResponse.minus(VectorFactory.getDefault().copyArray(
        new double[] {
            evComponent.getMean().doubleValue()
            }));
    final MultivariateGaussian posteriorState = updatedParticle.getLinearState().clone();
    filter.update(posteriorState, 
        diffAugResponse);
    updatedParticle.setLinearState(posteriorState);
    
    return updatedParticle;
  }


  
  
}
