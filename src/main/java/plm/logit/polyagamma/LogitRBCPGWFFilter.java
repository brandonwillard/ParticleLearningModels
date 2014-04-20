package plm.logit.polyagamma;

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
import com.statslibextensions.statistics.distribution.PolyaGammaDistribution;
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
public class LogitRBCPGWFFilter extends 
  AbstractParticleFilter<ObservedValue<Vector, Matrix>, LogitPGParticle> {

  public class LogitRBCWUpdater extends AbstractCloneableSerializable
      implements
        Updater<ObservedValue<Vector, Matrix>, LogitPGParticle> {

    final protected Random rng;
    final protected KalmanFilter initialFilter;
    final protected MultivariateGaussian initialPrior;

    public LogitRBCWUpdater(
        KalmanFilter initialFilter, 
        MultivariateGaussian initialPrior, 
        Random rng) {
      this.initialPrior = initialPrior;
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
    public double computeLogLikelihood(LogitPGParticle particle, 
        ObservedValue<Vector, Matrix> observation) {
      
      double logLikelihood = UnivariateGaussian.PDF.logEvaluate(
          particle.getAugResponseSample().getElement(0), 
          particle.getPriorPredMean(), particle.getPriorPredCov()); 
      
      return logLikelihood;
    }

    @Override
    public DataDistribution<LogitPGParticle> createInitialParticles(int numParticles) {

      final DataDistribution<LogitPGParticle> initialParticles =
          CountedDataDistribution.create(true);
      for (int i = 0; i < numParticles; i++) {
        
        final MultivariateGaussian initialPriorState = initialPrior.clone();
        final KalmanFilter kf = this.initialFilter.clone();
        
        final LogitPGParticle particleMvgDPDist =
            new LogitPGParticle(null, kf, initialPriorState);
        initialParticles.increment(particleMvgDPDist);
      }
      return initialParticles;
    }

    /**
     * In this model/filter, there's no need for blind samples from the predictive distribution.
     */
    @Override
    public LogitPGParticle update(LogitPGParticle previousParameter) {
      return previousParameter;
    }

  }

  final int K;
  final protected KalmanFilter initialFilter;
  
  /**
   * Estimate a dynamic logit model using water-filling using
   * Polya-Gamma latent variables.
   * 
   * @param initialPrior
   * @param F
   * @param G
   * @param modelCovariance
   * @param rng
   */
  public LogitRBCPGWFFilter(
      MultivariateGaussian initialPrior,
      Matrix F, Matrix G, Matrix  modelCovariance, 
      int K, Random rng) {
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
    this.setUpdater(new LogitRBCWUpdater(initialFilter, initialPrior, rng));
    this.setRandom(rng);
    this.K = K;
  }

  @Override
  public void update(DataDistribution<LogitPGParticle> target, ObservedValue<Vector, Matrix> data) {

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;

    /*
     * XXX: This treemap cannot be used for anything other than
     * what it's currently being used for, since the interface
     * contract is explicitly broken with our comparator.
     */
    TreeMap<Double, LogitPGParticle> particleTree = Maps.
        <Double, Double, LogitPGParticle>newTreeMap(
        new Comparator<Double>() {
          @Override
          public int compare(Double o1, Double o2) {
            return o1 < o2 ? 1 : -1;
          }
        });
    for (Entry<LogitPGParticle, ? extends Number> particleEntry : target.asMap().entrySet()) {
      final LogitPGParticle particle = particleEntry.getKey();

      final MultivariateGaussian predictivePrior = particle.getLinearState().clone();
      KalmanFilter kf = particle.getRegressionFilter();
      final Matrix G = kf.getModel().getA();
      final Matrix F = data.getObservedData();
      predictivePrior.setMean(G.times(predictivePrior.getMean()));
      predictivePrior.setCovariance(
          G.times(predictivePrior.getCovariance()).times(G.transpose())
            .plus(kf.getModelCovariance()));
      final Vector betaMean = predictivePrior.getMean();
      
      final double k_t = data.getObservedValue().getElement(0) - 1d/2d;

      final int particleCount;
      if (particleEntry.getValue() instanceof MutableDoubleCount) {
        particleCount = ((MutableDoubleCount)particleEntry.getValue()).count; 
      } else {
        particleCount = 1;
      }
      for (int p = 0; p < particleCount; p++) {

        for (int j = 0; j < K; j++) {

          /*
           * Sample and compute the latent variable...
           */
          final double omega = PolyaGammaDistribution.sample(0d, this.getRandom());
          final double z_t = k_t / omega;
          
          final LogitPGParticle predictiveParticle = particle.clone();
          predictiveParticle.setPreviousParticle(particle);
          predictiveParticle.setBetaSample(betaMean);
          predictiveParticle.setLinearState(predictivePrior);
          
          predictiveParticle.setAugResponseSample(
              VectorFactory.getDefault().copyValues(z_t));

          /*
           * Update the observed data for the regression component.
           */
          predictiveParticle.getRegressionFilter().getModel().setC(F);

          final Matrix compVar = MatrixFactory.getDefault().copyArray(
              new double[][] {{1d/omega}});
          predictiveParticle.getRegressionFilter().setMeasurementCovariance(compVar);
          
          final double compPredPriorObsMean = F.times(betaMean).getElement(0) ;
          final double compPredPriorObsCov = 
               F.times(predictivePrior.getCovariance()).times(F.transpose()).getElement(0, 0) 
               + 1d/omega;
          predictiveParticle.setPriorPredMean(compPredPriorObsMean);
          predictiveParticle.setPriorPredCov(compPredPriorObsCov);

          final double logLikelihood = 
              this.updater.computeLogLikelihood(predictiveParticle, data);

          // FIXME we're just assuming equivalent particles had equal weight
          final double priorLogWeight = particleEntry.getValue().doubleValue() 
              - Math.log(particleCount);
          
          final double jointLogLikelihood = logLikelihood + priorLogWeight;

          predictiveParticle.setLogWeight(jointLogLikelihood);

          particleTotalLogLikelihood = LogMath.add(particleTotalLogLikelihood, jointLogLikelihood);
          particleTree.put(jointLogLikelihood, predictiveParticle);
        }
      }
    }

    final CountedDataDistribution<LogitPGParticle> resampledParticles =
        ExtSamplingUtils.waterFillingResample( 
            Doubles.toArray(particleTree.keySet()), 
            particleTotalLogLikelihood,
            Lists.newArrayList(particleTree.values()), 
            random, this.numParticles);

    /*
     * Propagate
     */
    target.clear();
    for (final Entry<LogitPGParticle, ? extends Number> particleEntry : resampledParticles.asMap().entrySet()) {
      final LogitPGParticle updatedParticle = sufficientStatUpdate(
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

  private LogitPGParticle sufficientStatUpdate(
      LogitPGParticle priorParticle, ObservedValue<Vector, Matrix> data) {

    final LogitPGParticle updatedParticle = priorParticle.clone();
    final KalmanFilter filter = updatedParticle.getRegressionFilter(); 

    final MultivariateGaussian posteriorState = updatedParticle.getLinearState().clone();
    filter.update(posteriorState, updatedParticle.getAugResponseSample());

    updatedParticle.setLinearState(posteriorState);
    
    return updatedParticle;
  }


  
  
}
