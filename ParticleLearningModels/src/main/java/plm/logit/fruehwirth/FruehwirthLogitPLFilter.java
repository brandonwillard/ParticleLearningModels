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
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.TreeMap;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.common.collect.TreeMultimap;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.statistics.ExtSamplingUtils;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;
import com.statslibextensions.statistics.distribution.FruewirthSchnatterEV1Distribution;
import com.statslibextensions.statistics.distribution.WFCountedDataDistribution;
import com.statslibextensions.util.ExtStatisticsUtils;
import com.statslibextensions.util.ObservedValue;

/**
 * Particle filter implementing a logistic regression with water-filling.
 * The logistic follows Fruehwirth-Schnatter.
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
     * @param particle
     * @param observation
     * @return
     */
    @Override
    public double computeLogLikelihood(FruehwirthLogitParticle particle, 
        ObservedValue<Vector, Matrix> observation) {
      
      final UnivariateGaussian evComponent = particle.getEVcomponent();
      double logLikelihood = UnivariateGaussian.PDF.logEvaluate(
            particle.getAugResponseSample().getElement(0), 
            particle.getPriorPredMean(),
            particle.getPriorPredCov());
      
      return logLikelihood;
    }

    @Override
    public DataDistribution<FruehwirthLogitParticle> createInitialParticles(int numParticles) {

      final DataDistribution<FruehwirthLogitParticle> initialParticles =
          CountedDataDistribution.create(true);
      for (int i = 0; i < numParticles; i++) {
        
        final MultivariateGaussian initialPriorState = initialPrior.clone();
        final KalmanFilter kf = this.initialFilter.clone();
        final int componentId = this.rng.nextInt(10);
        final UnivariateGaussian evDist = this.evDistribution.
            getDistributions().get(componentId);
        
        final FruehwirthLogitParticle particleMvgDPDist =
            new FruehwirthLogitParticle(null, 
                kf, initialPriorState, evDist);
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

  protected final boolean waterFilling;

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
   * @param waterFilling use water-filling or not 
   * @param rng
   */
  public FruehwirthLogitPLFilter(
      MultivariateGaussian initialPrior,
      Matrix F, Matrix G, Matrix  modelCovariance, 
      boolean waterFilling, Random rng) {
    Preconditions.checkArgument(F.getNumRows() == 1);
    Preconditions.checkArgument(F.getNumColumns() == G.getNumRows());
    Preconditions.checkArgument(G.getNumColumns() == modelCovariance.getNumRows());
    this.waterFilling = waterFilling;
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

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;

    /*
     * XXX: This treemap cannot be used for anything other than
     * what it's currently being used for, since the interface
     * contract is explicitly broken with our comparator.
     */
    CountedDataDistribution<FruehwirthLogitParticle> expandedDist =
        CountedDataDistribution.create(true);
    TreeMap<Double, FruehwirthLogitParticle> particleTree = Maps.
        <Double, Double, FruehwirthLogitParticle>newTreeMap(
        new Comparator<Double>() {
          @Override
          public int compare(Double o1, Double o2) {
            return o1 < o2 ? 1 : -1;
          }
        });
//    TreeMultimap<FruehwirthLogitParticle, Double > particleTree = 
//        TreeMultimap.create(Ordering.arbitrary(), 
//            Ordering.natural().reverse());
    for (Entry<FruehwirthLogitParticle, ? extends Number> particleEntry : target.asMap().entrySet()) {
      final FruehwirthLogitParticle particle = particleEntry.getKey();

      /*
       * Fruewirth-Schnatter's method for sampling...
       */
      final MultivariateGaussian predictivePrior = particle.getLinearState().clone();
      KalmanFilter kf = particle.getRegressionFilter();
      final Matrix G = kf.getModel().getA();
      final Matrix F = data.getObservedData();
      predictivePrior.setMean(G.times(predictivePrior.getMean()));
      predictivePrior.setCovariance(
          G.times(predictivePrior.getCovariance()).times(G.transpose())
            .plus(kf.getModelCovariance()));
      final Vector betaSample = predictivePrior.getMean();
      final double predPriorObsMean = 
            F.times(betaSample).getElement(0);

      // X * beta
      final double lambda = Math.exp(predPriorObsMean);
      final int particleCount;
      if (particleEntry.getValue() instanceof MutableDoubleCount) {
        particleCount = ((MutableDoubleCount)particleEntry.getValue()).count; 
      } else {
        particleCount = 1;
      }
      for (int p = 0; p < particleCount; p++) {
        final double mlU = -Math.log(this.random.nextDouble());
        final boolean isOne = data.getObservedValue().getElement(0) > 0d;
        final double mlV = !isOne ? -Math.log(this.random.nextDouble()) : 0d;
        final double dSampledAugResponse = -Math.log(mlU/(1d+lambda) + mlV/lambda);

        Vector sampledAugResponse = VectorFactory.getDefault().copyValues(dSampledAugResponse);

        for (int j = 0; j < 10; j++) {
          final FruehwirthLogitParticle predictiveParticle = particle.clone();
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

              //this.updater.computeLogLikelihood(predictiveParticle, data)
          final double compLogLikelihood = UnivariateGaussian.PDF.logEvaluate(
                dSampledAugResponse, 
                compPredPriorObsMean,
                compPredPriorObsCov);

          // FIXME we're just assuming equivalent particles had equal weight
          final double priorLogWeight = particleEntry.getValue().doubleValue() 
              - Math.log(particleCount);
          
          final double logLikelihood = 
              compLogLikelihood
              + Math.log(this.evDistribution.getPriorWeights()[j])
              + priorLogWeight;

          predictiveParticle.setWeight(logLikelihood);

          particleTotalLogLikelihood = LogMath.add(particleTotalLogLikelihood, logLikelihood);
          particleTree.put(logLikelihood, predictiveParticle);
          expandedDist.increment(predictiveParticle, logLikelihood);
        }
      }
    }
    List<FruehwirthLogitParticle> samples = null;
    final CountedDataDistribution<FruehwirthLogitParticle> resampledParticles;
    if (this.waterFilling) {
      resampledParticles =
        ExtSamplingUtils.waterFillingResample( 
            Doubles.toArray(particleTree.keySet()), 
            particleTotalLogLikelihood,
            Lists.newArrayList(particleTree.values()), 
            random, this.numParticles);
    } else {
//      final double[] cumulativeLogWeights = ExtSamplingUtils.accumulate(particleTree.keySet());
//      final List<FruehwirthLogitParticle> supportObjs = Lists.newArrayList(particleTree.values());
//      samples = ExtSamplingUtils.
//          sampleReplaceCumulativeLogScale(
//            cumulativeLogWeights, supportObjs, 
//            random, this.numParticles);

//      samples = Lists.newArrayList();
//      while (samples.size() < this.numParticles) {
//        samples.add(particleTree.pollFirstEntry().getValue());
//      }

//      if (this.computeEffectiveParticles(target) < 0.95 * this.numParticles) {
        samples = expandedDist.sample(random, numParticles);
//      }
      resampledParticles =
          new CountedDataDistribution<FruehwirthLogitParticle>(true);
      resampledParticles.incrementAll(samples);

//      resampledParticles =
//          new CountedDataDistribution<FruehwirthLogitParticle>(true);
//      while (resampledParticles.getTotalCount() < this.numParticles) {
//        Entry<? extends Number, FruehwirthLogitParticle> e = particleTree.pollFirstEntry();
//        resampledParticles.increment(e.getValue(), e.getKey().doubleValue());
//      }
    }

    /*
     * Propagate
     */
    target.clear();
    for (final Entry<FruehwirthLogitParticle, ? extends Number> particleEntry : resampledParticles.asMap().entrySet()) {
      final FruehwirthLogitParticle updatedParticle = sufficientStatUpdate(
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

  private FruehwirthLogitParticle sufficientStatUpdate(
      FruehwirthLogitParticle priorParticle, ObservedValue<Vector, Matrix> data) {
    final FruehwirthLogitParticle updatedParticle = priorParticle.clone();
    
    final Vector sampledAugResponse = priorParticle.getAugResponseSample();
    final KalmanFilter filter = updatedParticle.getRegressionFilter(); 
    final UnivariateGaussian evComponent = updatedParticle.EVcomponent;
    // TODO we should've already set this, so it might be redundant.
    filter.setMeasurementCovariance(
        MatrixFactory.getDefault().copyArray(new double[][] {{
          evComponent.getVariance()}}));

    final Vector diffAugResponse = 
        sampledAugResponse.minus(VectorFactory.getDefault().copyArray(
        new double[] {
            evComponent.getMean().doubleValue()
            }));
    final MultivariateGaussian posteriorState = updatedParticle.getLinearState().clone();
    filter.update(posteriorState, diffAugResponse);
    updatedParticle.setLinearState(posteriorState);
    
    return updatedParticle;
  }


  
  
}
