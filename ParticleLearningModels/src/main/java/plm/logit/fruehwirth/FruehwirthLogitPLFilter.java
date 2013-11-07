package plm.logit.fruehwirth;

import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
import gov.sandia.cognition.statistics.UnivariateDistribution;
import gov.sandia.cognition.statistics.bayesian.AbstractParticleFilter;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.MultivariateStudentTDistribution;
import gov.sandia.cognition.statistics.distribution.NormalInverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.ObjectUtil;

import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.TreeMap;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.primitives.Doubles;
import com.statslibextensions.statistics.CountedDataDistribution;
import com.statslibextensions.statistics.ExtSamplingUtils;
import com.statslibextensions.statistics.bayesian.DlmUtils;
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

    public FruehwirthLogitPLUpdater(
        KalmanFilter initialFilter, 
        MultivariateGaussian initialPrior, FruewirthSchnatterEV1Distribution evDistribution, Random rng) {
      this.initialPrior = initialPrior;
      this.evDistribution = evDistribution;
      this.initialFilter = initialFilter;
      this.rng = rng;
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

      final MultivariateGaussian predictivePrior = particle.getLinearState().clone();
      KalmanFilter kf = particle.getRegressionFilter();
      kf.predict(predictivePrior);
      final Matrix F = kf.getModel().getC();

      final double predPriorObsMean = F.times(predictivePrior.getMean()).getElement(0)
          + particle.getEVcomponent().getMean();
      final double predPriorObsCov = F.times(predictivePrior.getCovariance()).times(F.transpose())
          .plus(kf.getMeasurementCovariance()).getElement(0, 0);
      
      particle.setPriorPredMean(predPriorObsMean);
      particle.setPriorPredCov(predPriorObsCov);


      final double likelihood = observation.getObservedValue().getElement(0) > 0 ?
          1d - UnivariateGaussian.CDF.evaluate(0d, predPriorObsMean, predPriorObsCov)
              : UnivariateGaussian.CDF.evaluate(0d, predPriorObsMean, predPriorObsCov); 
          
      return Math.log(likelihood);
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

  final protected FruewirthSchnatterEV1Distribution evDistribution = 
      new FruewirthSchnatterEV1Distribution();
  final protected KalmanFilter initialFilter;
  
  public FruehwirthLogitPLFilter(
      MultivariateGaussian initialPrior,
      Matrix F, Matrix G, Matrix  modelCovariance, Random rng) {
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
  public void update(DataDistribution<FruehwirthLogitParticle> target, ObservedValue<Vector, Matrix> data) {
    Preconditions.checkState(target.getDomainSize() == this.numParticles);

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
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
      for (int j = 0; j < 10; j++) {
        /*
         * TODO could avoid cloning if we didn't change the measurement covariance,
         * but instead used the componentDist explicitly.
         */
        final FruehwirthLogitParticle predictiveParticle = particle.clone();
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
            + particleEntry.getValue().doubleValue();
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
    
    final double f = updatedParticle.getPriorPredMean();
    final double Q = updatedParticle.getPriorPredCov();
    final double U = this.random.nextDouble();
    
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
    final double PhiUpper = UnivariateGaussian.CDF.evaluate(limUpper, f, Q);
    final double PhiLower = UnivariateGaussian.CDF.evaluate(limLower, f, Q);
    final double dSampledAugResponse = UnivariateGaussian.CDF.Inverse.evaluate(
        PhiLower + U * (PhiUpper - PhiLower), 0, 1)
        *Math.sqrt(Q) + f;
    final Vector sampledAugResponse = VectorFactory.getDefault().copyValues(dSampledAugResponse);

    /*
     * Let's try Fruewirth-Schnatter's method for sampling...
     */
//    final Vector betaSample = updatedParticle.linearState.sample(this.random);
//
//    // X * beta
//    final double lambda = Math.exp(data.getObservedData().times(betaSample).getElement(0));
//    final double dSampledAugResponse = -Math.log(
//        -Math.log(this.random.nextDouble())/(1d+lambda)
//        - (data.getObservedValue().getElement(0) > 0d 
//            ? 0d : Math.log(this.random.nextDouble())/lambda));
//
//    final Vector sampledAugResponse = VectorFactory.getDefault().copyValues(dSampledAugResponse);

    updatedParticle.setAugResponseSample(sampledAugResponse);

    // TODO we should've already set this, so it might be redundant.
    updatedParticle.regressionFilter.setMeasurementCovariance(MatrixFactory.getDefault().copyArray(
        new double[][] {{updatedParticle.EVcomponent.getVariance()}}));

    final MultivariateGaussian posteriorState = updatedParticle.linearState;
    updatedParticle.getRegressionFilter().update(posteriorState, sampledAugResponse);
    
    return updatedParticle;
  }

  public List<DataDistribution<FruehwirthLogitParticle>> batchUpdate(
      List<ObservedValue<Vector, Matrix>> data) {

    return null;
  }
  
  
}
