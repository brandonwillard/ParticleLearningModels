package plm.logit.fruehwirth;

import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.math.LogNumber;
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

import java.lang.management.ThreadInfo;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.primitives.Doubles;
import com.higherfrequencytrading.affinity.AffinityStrategies;
import com.higherfrequencytrading.affinity.AffinityThreadFactory;
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
public class LogitParRBCWFFilter extends AbstractParticleFilter<ObservedValue<Vector, Matrix>, LogitMixParticle> {

  public class LogitParRBCWFUpdater extends AbstractCloneableSerializable
      implements
        Updater<ObservedValue<Vector, Matrix>, LogitMixParticle> {

    final protected Random rng;
    final protected FruewirthSchnatterEV1Distribution evDistribution;
    final protected KalmanFilter initialFilter;
    final protected MultivariateGaussian initialPrior;

    public LogitParRBCWFUpdater(
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
        if (observation.getObservedValue().getElement(0) > 0d) {
//        if (!observation.getObservedValue().isZero()) {
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
  
  final protected ExecutorService threadPool = Executors.newFixedThreadPool(
      Runtime.getRuntime().availableProcessors(), new AffinityThreadFactory("test-atf", 
          AffinityStrategies.DIFFERENT_CORE));
  
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
  public LogitParRBCWFFilter(
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
    this.setUpdater(new LogitParRBCWFUpdater(initialFilter, initialPrior,
        evDistribution, rng));
    this.setRandom(rng);
  }

  private class PropagateParticleTask implements Runnable {
    
    final private ObservedValue<Vector, Matrix> data;
    final private Entry<LogitMixParticle, ? extends Number> particleEntry;
    final private LogitParRBCWFFilter filter;
    final private ConcurrentMap<LogitMixParticle, Number> results;
    public LogitMixParticle resultParticle;

    public PropagateParticleTask(Entry<LogitMixParticle, ? extends Number> particleEntry,
      ObservedValue<Vector, Matrix> data, LogitParRBCWFFilter filter,
      ConcurrentMap<LogitMixParticle, Number> results) {
      this.particleEntry = particleEntry;
      this.data = data;
      this.filter = filter;
      this.results = results;
    }

    @Override
    public void run() {
      this.resultParticle = sufficientStatUpdate(
          particleEntry.getKey(), data);
      this.results.put(resultParticle, (Number)particleEntry.getValue());
    }
  }

  private class PrepAndWeightParticles implements Runnable {
    
    final private ObservedValue<Vector, Matrix> data;
    final private Entry<LogitMixParticle, ? extends Number> particleEntry;
    final private LogitParRBCWFFilter filter;
    final private ConcurrentMap<Double, LogitMixParticle> particleTree;
    public LogNumber particleTotalLogLikelihood;

    public PrepAndWeightParticles(Entry<LogitMixParticle, ? extends Number> particleEntry,
      ObservedValue<Vector, Matrix> data, LogitParRBCWFFilter filter,
      ConcurrentMap<Double, LogitMixParticle> particleTree, 
      LogNumber particleTotalLogLikelihood) {
      this.particleEntry = particleEntry;
      this.data = data;
      this.filter = filter;
      this.particleTree = particleTree;
      this.particleTotalLogLikelihood = particleTotalLogLikelihood;
    }

    @Override
    public void run() {
      double sourceTotalLogLikelihood = Double.NEGATIVE_INFINITY;

      final LogitMixParticle particle = particleEntry.getKey();

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
          final LogitMixParticle predictiveParticle = 
              particle.clone();

          predictiveParticle.setPreviousParticle(particle);
          predictiveParticle.setBetaSample(betaMean);
          predictiveParticle.setLinearState(predictivePrior);

          final UnivariateGaussian componentDist = 
              filter.evDistribution.getDistributions().get(j);

          predictiveParticle.setEVcomponent(componentDist);
          
          /*
           * Update the observed data for the regression component.
           */
          predictiveParticle.getRegressionFilter().getModel().setC(F);

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
              filter.updater.computeLogLikelihood(predictiveParticle, data);

          // FIXME we're just assuming equivalent particles had equal weight
          final double priorLogWeight = particleEntry.getValue().doubleValue() 
              - Math.log(particleCount);
          
          final double jointLogLikelihood = 
              logLikelihood
              // add the weight for this component
              + Math.log(filter.evDistribution.getPriorWeights()[j])
              + priorLogWeight;

          predictiveParticle.setWeight(jointLogLikelihood);

          sourceTotalLogLikelihood = LogMath.add(sourceTotalLogLikelihood, jointLogLikelihood);
          particleTree.put(jointLogLikelihood, predictiveParticle);
        }
      }
      /*
       * particleTotalLogLikelihood.plusEquals is sync'ed, so we do this
       * as little as possible.
       */
      LogNumber thisLikelihood = new LogNumber();
      thisLikelihood.setLogValue(sourceTotalLogLikelihood);
      particleTotalLogLikelihood.plusEquals(thisLikelihood);
    }
  }
  
  /**
   * This is only thread safe for our exact purposes...
   * 
   * @author bwillar0
   */
  public static class SafeLogNumber extends LogNumber {

    private static final long serialVersionUID = 3780756110537246803L;

    @Override
    synchronized public void plusEquals(LogNumber other) {
      super.plusEquals(other);
    }
  }

  @Override
  public void update(DataDistribution<LogitMixParticle> target, ObservedValue<Vector, Matrix> data) {

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    SafeLogNumber particleTotalLogLikelihood = new SafeLogNumber();

    /*
     * XXX: This treemap cannot be used for anything other than
     * what it's currently being used for, since the interface
     * contract is explicitly broken with our comparator.
     */
    ConcurrentSkipListMap<Double, LogitMixParticle> particleTree = 
        new ConcurrentSkipListMap<Double, LogitMixParticle>(
        new Comparator<Double>() {
          @Override
          public int compare(Double o1, Double o2) {
            return o1 < o2 ? 1 : -1;
          }
        });

    List<Callable<Object>> tasks = Lists.newArrayList();
    /*
     * Create tasks and wait until they're finished.
     */
    for (Entry<LogitMixParticle, ? extends Number> particleEntry : target.asMap().entrySet()) {
      final PrepAndWeightParticles newParticleTask = new PrepAndWeightParticles(particleEntry, 
          data, this, particleTree, particleTotalLogLikelihood);
      tasks.add(Executors.callable(newParticleTask));
    }
    
    try {
      threadPool.invokeAll(tasks);
    } catch (InterruptedException e1) {
      e1.printStackTrace();
    }

    final CountedDataDistribution<LogitMixParticle> resampledParticles =
        ExtSamplingUtils.waterFillingResample( 
            Doubles.toArray(particleTree.keySet()), 
            particleTotalLogLikelihood.getLogValue(),
            Lists.newArrayList(particleTree.values()), 
            random, this.numParticles);

    /*
     * Propagate
     */
    ConcurrentMap<LogitMixParticle, Number> propagatedParticles = 
        Maps.newConcurrentMap();
    for (final Entry<LogitMixParticle, ? extends Number> particleEntry : resampledParticles.asMap().entrySet()) {
      PropagateParticleTask newParticleTask = new PropagateParticleTask(particleEntry, 
          data, this, propagatedParticles);
      tasks.add(Executors.callable(newParticleTask));
    }
    try {
      threadPool.invokeAll(tasks);
    } catch (InterruptedException e1) {
      e1.printStackTrace();
    }

    target.clear();
    for (Entry<LogitMixParticle, ? extends Number> particleEntry : propagatedParticles.entrySet()) {
      if (particleEntry.getValue() instanceof MutableDoubleCount) {
        ((CountedDataDistribution)target).set(particleEntry.getKey(), 
            particleEntry.getValue().doubleValue(), 
            ((MutableDoubleCount)particleEntry.getValue()).count); 
      } else {
        target.set(particleEntry.getKey(), particleEntry.getValue().doubleValue());
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
      dSampledAugResponse = ExtSamplingUtils.truncNormalSampleRej(
          getRandom(), 
          0d, Double.POSITIVE_INFINITY, tdistMean, tdistVar); 
    } else {
      // Sample (-Inf, 0)
      dSampledAugResponse = ExtSamplingUtils.truncNormalSampleRej(
          getRandom(), 
          Double.NEGATIVE_INFINITY, 0d, tdistMean, tdistVar); 
    }
    
    Preconditions.checkState(Doubles.isFinite(dSampledAugResponse));

    return dSampledAugResponse;
  }

  private LogitMixParticle sufficientStatUpdate(
      LogitMixParticle priorParticle, ObservedValue<Vector, Matrix> data) {

    final LogitMixParticle updatedParticle = priorParticle.clone();
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
