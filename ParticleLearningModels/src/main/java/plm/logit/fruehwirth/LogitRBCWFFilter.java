package plm.logit.fruehwirth;

import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
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
public class LogitRBCWFFilter extends AbstractParticleFilter<ObservedValue<Vector, Matrix>, LogitFSParticle> {

  public class FruehwirthLogitPLUpdater extends AbstractCloneableSerializable
      implements
        Updater<ObservedValue<Vector, Matrix>, LogitFSParticle> {

    final protected Random rng;
    final protected FruewirthSchnatterEV1Distribution evDistribution;
    final protected KalmanFilter initialFilter;
    final protected MultivariateGaussian initialPrior;

    public FruehwirthLogitPLUpdater(
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
    public double computeLogLikelihood(LogitFSParticle particle, 
        ObservedValue<Vector, Matrix> observation) {
      
      // The prior pred. dist should already be conditional on the current component...
      final double upperMean = particle.getPriorPredMean();
      final double upperVar = particle.getPriorPredCov();
      double logLikelihood = Double.NEGATIVE_INFINITY; 
      double[] componentLikelihoods = new double[10];
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
        
        componentLikelihoods[i] = Math.exp(partLogLik);
        
        logLikelihood = LogMath.add(logLikelihood, partLogLik);
      }
      
      particle.setComponentLikelihoods(componentLikelihoods);
      
      return logLikelihood;
    }

    @Override
    public DataDistribution<LogitFSParticle> createInitialParticles(int numParticles) {

      final DataDistribution<LogitFSParticle> initialParticles =
          CountedDataDistribution.create(true);
      for (int i = 0; i < numParticles; i++) {
        
        final MultivariateGaussian initialPriorState = initialPrior.clone();
        final KalmanFilter kf = this.initialFilter.clone();
        final int componentId = this.rng.nextInt(10);
        final UnivariateGaussian evDist = this.evDistribution.
            getDistributions().get(componentId);
        
        final LogitFSParticle particleMvgDPDist =
            new LogitFSParticle(null, 
                kf, initialPriorState, evDist);
        initialParticles.increment(particleMvgDPDist);
      }
      return initialParticles;
    }

    /**
     * In this model/filter, there's no need for blind samples from the predictive distribution.
     */
    @Override
    public LogitFSParticle update(LogitFSParticle previousParameter) {
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
  public LogitRBCWFFilter(
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
    this.setUpdater(new FruehwirthLogitPLUpdater(initialFilter, initialPrior,
        evDistribution, rng));
    this.setRandom(rng);
  }

  @Override
  public void update(DataDistribution<LogitFSParticle> target, ObservedValue<Vector, Matrix> data) {

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;

    /*
     * XXX: This treemap cannot be used for anything other than
     * what it's currently being used for, since the interface
     * contract is explicitly broken with our comparator.
     */
    TreeMap<Double, LogitFSParticle> particleTree = Maps.
        <Double, Double, LogitFSParticle>newTreeMap(
        new Comparator<Double>() {
          @Override
          public int compare(Double o1, Double o2) {
            return o1 < o2 ? 1 : -1;
          }
        });
    for (Entry<LogitFSParticle, ? extends Number> particleEntry : target.asMap().entrySet()) {
      final LogitFSParticle particle = particleEntry.getKey();

      final MultivariateGaussian predictivePrior = particle.getLinearState().clone();
      KalmanFilter kf = particle.getRegressionFilter();
      final Matrix G = kf.getModel().getA();
      final Matrix F = data.getObservedData();
      predictivePrior.setMean(G.times(predictivePrior.getMean()));
      predictivePrior.setCovariance(
          G.times(predictivePrior.getCovariance()).times(G.transpose())
            .plus(kf.getModelCovariance()));
      final Vector betaMean = predictivePrior.getMean();

      final int particleCount;
      if (particleEntry.getValue() instanceof MutableDoubleCount) {
        particleCount = ((MutableDoubleCount)particleEntry.getValue()).count; 
      } else {
        particleCount = 1;
      }
      for (int p = 0; p < particleCount; p++) {

        for (int j = 0; j < 10; j++) {
          final LogitFSParticle predictiveParticle = particle.clone();
          predictiveParticle.setPreviousParticle(particle);
          predictiveParticle.setBetaSample(betaMean);
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
               F.times(betaMean).getElement(0) 
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

    final CountedDataDistribution<LogitFSParticle> resampledParticles =
        ExtSamplingUtils.waterFillingResample( 
            Doubles.toArray(particleTree.keySet()), 
            particleTotalLogLikelihood,
            Lists.newArrayList(particleTree.values()), 
            random, this.numParticles);

    /*
     * Propagate
     */
    target.clear();
    for (final Entry<LogitFSParticle, ? extends Number> particleEntry : resampledParticles.asMap().entrySet()) {
      final LogitFSParticle updatedParticle = sufficientStatUpdate(
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

  /**
   * Sample the upper augmented response variable for a difference of 
   * EV(0,1) distributions given an observation.</br>
   * TODO, FIXME: Uses the inverse CDF, so it's not the best/quickest...
   * @param upperMean
   * @param upperVar
   * @param isOne
   * @return upper utility sample
   */
  private double sampleAugResponse(double upperMean, 
    double upperVar, boolean isOne, UnivariateGaussian partComponent) {

    final double tdistMean = upperMean - partComponent.getMean();
    final double tdistVar = upperVar + partComponent.getVariance();

    final double dSampledAugResponse;  
    if (isOne) {
      // Sample [0, Inf)
      dSampledAugResponse = ExtSamplingUtils.truncNormalSampleRej(getRandom(), 
          0d, Double.POSITIVE_INFINITY, tdistMean, tdistVar); 
    } else {
      // Sample (-Inf, 0)
      dSampledAugResponse = ExtSamplingUtils.truncNormalSampleRej(getRandom(), 
          Double.NEGATIVE_INFINITY, 0d, tdistMean, tdistVar); 
    }
    
    Preconditions.checkState(Doubles.isFinite(dSampledAugResponse));

    return dSampledAugResponse;
  }

  private LogitFSParticle sufficientStatUpdate(
      LogitFSParticle priorParticle, ObservedValue<Vector, Matrix> data) {

    final LogitFSParticle updatedParticle = priorParticle.clone();
    final KalmanFilter filter = updatedParticle.getRegressionFilter(); 

    final UnivariateGaussian evComponent = updatedParticle.EVcomponent;

    final boolean isOne = !data.getObservedValue().isZero();

    final int smplLowerIdx = DiscreteSamplingUtil.sampleIndexFromProportions(
        getRandom(), updatedParticle.getComponentLikelihoods());
//    final int smplLowerIdx = DiscreteSamplingUtil.sampleIndexFromProbabilities(
//        getRandom(), this.evDistribution.getPriorWeights());

    final UnivariateGaussian partComponent = 
        this.evDistribution.getDistributions().get(smplLowerIdx);

    final double dsampledAugResponse = sampleAugResponse(
        updatedParticle.getPriorPredMean(), 
        updatedParticle.getPriorPredCov(), isOne,
        partComponent);

    // TODO we should've already set this, so it might be redundant.
    filter.setMeasurementCovariance(
        MatrixFactory.getDefault().copyArray(new double[][] {{
          evComponent.getVariance() + partComponent.getVariance()}}));

    final Vector sampledAugResponse = 
        VectorFactory.getDefault().copyValues(
            dsampledAugResponse 
            - evComponent.getMean().doubleValue()
            - partComponent.getMean().doubleValue());

    updatedParticle.setAugResponseSample(sampledAugResponse); 

    final MultivariateGaussian posteriorState = updatedParticle.getLinearState().clone();
    filter.update(posteriorState, sampledAugResponse);

    updatedParticle.setLinearState(posteriorState);
    
    return updatedParticle;
  }


  
  
}
