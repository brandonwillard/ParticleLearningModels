package plm.logit.fruehwirth;

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
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.TreeMap;

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
 * A Particle Learning filter for a multivariate Gaussian Dirichlet Process.
 * 
 * @author bwillard
 * 
 */
public class FruehwirthLogitPLFilter extends AbstractParticleFilter<ObservedValue<Vector, Matrix>, FruehwirthLogitParticle> {

  public class FruehwirthLogitPLUpdater extends AbstractCloneableSerializable
      implements
        Updater<ObservedValue<Vector, Matrix>, FruehwirthLogitParticle> {

    final protected Random rng;
    final protected FruewirthSchnatterEV1Distribution evDistribution;
    final protected KalmanFilter initialFilter;
    final protected MultivariateGaussian initialPrior;
    final protected boolean fsResponseSampling;

    public FruehwirthLogitPLUpdater(
        KalmanFilter initialFilter, 
        MultivariateGaussian initialPrior, 
        FruewirthSchnatterEV1Distribution evDistribution, 
        boolean fsResponseSampling, Random rng) {
      this.initialPrior = initialPrior;
      this.evDistribution = evDistribution;
      this.initialFilter = initialFilter;
      this.rng = rng;
      this.fsResponseSampling = fsResponseSampling;
    }

    /**
     * 
     * @param particle
     * @param observation
     * @return
     */
    @Override
    public double computeLogLikelihood(FruehwirthLogitParticle particle, 
        ObservedValue<Vector, Matrix> observation) {
      
      /*
       * TODO when using FS aug sampling, we should probably go all the way
       * and replicate their beta sampling.
       * That would require a change here...
       */
      final MultivariateGaussian predictivePrior = particle.getLinearState().clone();
      KalmanFilter kf = particle.getRegressionFilter();
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

      double logLikelihood;
      if (!this.fsResponseSampling) {
        logLikelihood = observation.getObservedValue().getElement(0) > 0 ?
            1d - UnivariateGaussian.CDF.evaluate(0d, predPriorObsMean, predPriorObsCov)
                : UnivariateGaussian.CDF.evaluate(0d, predPriorObsMean, predPriorObsCov); 
        logLikelihood = Math.log(logLikelihood);
      } else {
        logLikelihood = UnivariateGaussian.PDF.logEvaluate(
            particle.getAugResponseSample().getElement(0), 
            predPriorObsMean, predPriorObsCov);
      }
      
      return logLikelihood;
    }

    @Override
    public DataDistribution<FruehwirthLogitParticle> createInitialParticles(int numParticles) {

      final DataDistribution<FruehwirthLogitParticle> initialParticles =
          CountedDataDistribution.create(true);
      for (int i = 0; i < numParticles; i++) {
        
        final MultivariateGaussian initialPriorState = initialPrior.clone();
        final KalmanFilter kf = this.initialFilter.clone();
        
        final FruehwirthLogitParticle particleMvgDPDist =
            new FruehwirthLogitParticle(null, 
                kf, initialPriorState, null);
        initialParticles.increment(particleMvgDPDist);
      }
      return initialParticles;
    }

    /**
     * In this model/filter, there's no need for blind samples from the predictive distribution.
     */
    @Override
    public FruehwirthLogitParticle update(FruehwirthLogitParticle previousParameter) {
      return previousParameter;
    }

  }

  protected final boolean fsResponseSampling;

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
   * @param fsResponseSampling determines if Fruehwirth-Schnatter's method of augmented response
   * sampling should be used, or Rao-Blackwellization and sampling.
   */
  public FruehwirthLogitPLFilter(
      MultivariateGaussian initialPrior,
      Matrix F, Matrix G, Matrix  modelCovariance, Random rng, boolean fsResponseSampling) {
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
    this.fsResponseSampling = fsResponseSampling;
    this.setUpdater(new FruehwirthLogitPLUpdater(initialFilter, initialPrior,
        evDistribution, fsResponseSampling, rng));
    this.setRandom(rng);
  }

  @Override
  public void update(DataDistribution<FruehwirthLogitParticle> target, ObservedValue<Vector, Matrix> data) {
    Preconditions.checkState(target.getDomainSize() == this.numParticles);

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final double prevTotalLogLikelihood = target.getTotal();
//    final double[] logLikelihoods = new double[this.numParticles * 10];
//    final List<FruehwirthLogitParticle> particleSupport = Lists.newArrayListWithExpectedSize(this.numParticles * 10);
    /*
     * TODO: I like the idea of this tree-map, but not the
     * equality killing comparator.  This probably kills
     * the efficiency of the hashing, but i suppose we don't 
     * actually use that here!
     */
    TreeMap<Double, FruehwirthLogitParticle> particleTree = Maps.
        <Double, Double, FruehwirthLogitParticle>newTreeMap(
        new Comparator<Double>() {
          @Override
          public int compare(Double o1, Double o2) {
            return o1 < o2 ? 1 : -1;
          }
        });
//    int i = 0;
    for (Entry<FruehwirthLogitParticle, ? extends Number> particleEntry : target.asMap().entrySet()) {
      final FruehwirthLogitParticle particle = particleEntry.getKey();

      Vector sampledAugResponse = null;
      if (this.fsResponseSampling) {
        /*
         * Let's try Fruewirth-Schnatter's method for sampling...
         * TODO when using FS aug sampling, we should probably go all the way
         * and replicate their beta sampling.
         * That would require a change here...
         */
//        final Vector betaSample = particle.getLinearState().sample(this.random);
        final MultivariateGaussian predictivePrior = particle.getLinearState().clone();
        KalmanFilter kf = particle.getRegressionFilter();
        final Matrix G = kf.getModel().getA();
        predictivePrior.setMean(G.times(predictivePrior.getMean()));
        predictivePrior.setCovariance(
            G.times(predictivePrior.getCovariance()).times(G.transpose())
              .plus(kf.getModelCovariance()));
  
        // X * beta
        final double lambda = Math.exp(data.getObservedData().times(
            predictivePrior.getMean()).getElement(0));
        final double dSampledAugResponse = -Math.log(
            -Math.log(this.random.nextDouble())/(1d+lambda)
            - (data.getObservedValue().getElement(0) > 0d 
                ? 0d : Math.log(this.random.nextDouble())/lambda));
  
        sampledAugResponse = VectorFactory.getDefault().copyValues(dSampledAugResponse);
      }

      for (int j = 0; j < 10; j++) {
        /*
         * TODO could avoid cloning if we didn't change the measurement covariance,
         * but instead used the componentDist explicitly.
         */
        final FruehwirthLogitParticle predictiveParticle = particle.clone();
        
        if (this.fsResponseSampling)
          predictiveParticle.setAugResponseSample(sampledAugResponse); 

        final UnivariateGaussian componentDist = 
            this.evDistribution.getDistributions().get(j);

        predictiveParticle.setEVcomponent(componentDist);
        
        /*
         * Update the observed data for the regression component.
         */
        predictiveParticle.getRegressionFilter().getModel().setC(data.getObservedData());

        final Matrix compVar = MatrixFactory.getDefault().copyArray(
            new double[][] {{componentDist.getVariance()}});
        predictiveParticle.getRegressionFilter().setMeasurementCovariance(compVar);
        
        final double logLikelihood = this.updater.computeLogLikelihood(predictiveParticle, data)
            + Math.log(this.evDistribution.getPriorWeights()[j])
            + (particleEntry.getValue().doubleValue() - prevTotalLogLikelihood);

        particleTotalLogLikelihood = LogMath.add(particleTotalLogLikelihood, logLikelihood);
//        logLikelihoods[i] = logLikelihood;
//        particleSupport.add(predictiveParticle);
        particleTree.put(logLikelihood, predictiveParticle);
//        i++;
      }
    }

//    final WFCountedDataDistribution<FruehwirthLogitParticle> resampledParticles =
//        ExtSamplingUtils.waterFillingResample(logLikelihoods, particleTotalLogLikelihood,
//            particleSupport, random, this.numParticles);

    final WFCountedDataDistribution<FruehwirthLogitParticle> resampledParticles =
        ExtSamplingUtils.waterFillingResample(
            Doubles.toArray(particleTree.keySet()), 
            particleTotalLogLikelihood,
            Lists.newArrayList(particleTree.values()), 
            random, this.numParticles);

    /*
     * Propagate
     */
    target.clear();
    for (final Entry<FruehwirthLogitParticle, MutableDouble> particleEntry : resampledParticles.asMap().entrySet()) {
      final FruehwirthLogitParticle updatedParticle = sufficientStatUpdate(particleEntry.getKey(), data);
      target.set(updatedParticle, particleEntry.getValue().value);
    }
    Preconditions.checkState(target.getDomainSize() == this.numParticles);

  }

  private FruehwirthLogitParticle sufficientStatUpdate(
      FruehwirthLogitParticle priorParticle, ObservedValue<Vector, Matrix> data) {
    final FruehwirthLogitParticle updatedParticle = priorParticle.clone();
    
    final Vector sampledAugResponse;
    if (!this.fsResponseSampling) {
      /*
       * TODO why not just sample the half-normal every time and negate when
       * necessary?
       */
      final double limUpper;
      final double limLower;
      if (data.getObservedValue().getElement(0) > 0d) {
        limUpper = Double.POSITIVE_INFINITY;
        limLower = 0d;
      } else {
        limUpper = 0d;
        limLower = Double.NEGATIVE_INFINITY;
      }
      final double f = updatedParticle.getPriorPredMean();
      final double Q = updatedParticle.getPriorPredCov();
      final double PhiUpper = UnivariateGaussian.CDF.evaluate(limUpper, f, Q);
      final double PhiLower = UnivariateGaussian.CDF.evaluate(limLower, f, Q);
      final double U = this.random.nextDouble();
      final double dSampledAugResponse = UnivariateGaussian.CDF.Inverse.evaluate(
          PhiLower + U * (PhiUpper - PhiLower), 0, 1)
          *Math.sqrt(Q) + f;
      sampledAugResponse = VectorFactory.getDefault().copyValues(dSampledAugResponse);

      updatedParticle.setAugResponseSample(sampledAugResponse);
    } else {
      sampledAugResponse = priorParticle.getAugResponseSample();
    }

    final KalmanFilter filter = updatedParticle.getRegressionFilter(); 
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
