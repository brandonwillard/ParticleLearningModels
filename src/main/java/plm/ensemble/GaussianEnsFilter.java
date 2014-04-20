package plm.ensemble;

import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.AbstractParticleFilter;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.MultivariateStudentTDistribution;
import gov.sandia.cognition.util.AbstractCloneableSerializable;

import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.log4j.Logger;

import plm.ensemble.GaussianEnsParticle;
import plm.hmm.HmmTransitionState.ResampleType;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.ExtLogMath;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;
import com.statslibextensions.statistics.distribution.WFCountedDataDistribution;
import com.statslibextensions.util.ExtSamplingUtils;
import com.statslibextensions.util.ObservedValue;

/**
 * 
 * A simple ensemble particle filter.
 * 
 * @author bwillard
 * 
 */
public class GaussianEnsFilter extends AbstractParticleFilter<ObservedValue<Vector, ?>, 
  GaussianEnsParticle> {
  public class GaussianEnsUpdater extends AbstractCloneableSerializable implements
      Updater<ObservedValue<Vector, ?>, GaussianEnsParticle> {

    final protected MultivariateGaussian initialStatePrior;

    final protected KalmanFilter initialKf;
    final protected Random rng;
    
    public GaussianEnsUpdater(KalmanFilter kf, 
        MultivariateGaussian priorState, Random rng) {
      this.rng = rng;
      this.initialKf = kf;
      this.initialStatePrior = priorState;
    }

    @Override
    public double computeLogLikelihood(
      GaussianEnsParticle transParticle,
      ObservedValue<Vector,?> observation) {

      return Double.NaN;
    }

    @Override
    public DataDistribution<GaussianEnsParticle> createInitialParticles(
        int numParticles) {
      final CountedDataDistribution<GaussianEnsParticle> initialParticles =
          CountedDataDistribution.create(numParticles, true);
      for (int i = 0; i < numParticles; i++) {

        final KalmanFilter thisKf = this.initialKf.clone();

        final GaussianEnsParticle particle =
            null;
//            new GaussianEnsParticle(thisKf, 
//                ObservedValue.<Vector>create(0, null)
//                );
        
        particle.setResampleType(ResampleType.NONE);

        final double logWeight = -Math.log(numParticles);
        particle.setStateLogWeight(logWeight);
        initialParticles.increment(particle, logWeight);
      }
      return initialParticles;
    }

    @Override
    public GaussianEnsParticle update(
      GaussianEnsParticle predState) {

      final MultivariateGaussian posteriorState = predState.getState().clone();
      
      return null;
    }

  }

  final Logger log = Logger.getLogger(GaussianEnsFilter.class);
  final int numSubSamples;

  public GaussianEnsFilter(KalmanFilter kf, 
      MultivariateGaussian priorState,
      Random rng, int numSubSamples) {
    this.numSubSamples = numSubSamples;
    this.setUpdater(new GaussianEnsUpdater(kf, priorState, rng));
    this.setRandom(rng);
  }

  protected GaussianEnsParticle forwardParticle(
      GaussianEnsParticle prevState, ObservedValue<Vector,?> data) {
    
    return null;
  }

  /**
   * This update will split each particle off into K sub-samples, then water-fill.
   */
  @Override
  public void update(
    DataDistribution<GaussianEnsParticle> target,
    ObservedValue<Vector, ?> data) {

    /*
     * Propagate and compute prior predictive log likelihoods.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final List<Double> logLikelihoods = Lists.newArrayList();
    final List<GaussianEnsParticle> particleSupport =
        Lists.newArrayList();
    for (final GaussianEnsParticle particle : target
        .getDomain()) {

      final int particleCount =
          ((CountedDataDistribution) target).getCount(particle);

      final double particlePriorLogLik =
          target.getLogFraction(particle);

      for (int i = 0; i < this.numSubSamples * particleCount; i++) {

        /*
         * K many sub-samples of x_{t-1}  
         */

        final GaussianEnsParticle transStateTmp = 
            null;
//            new GaussianEnsParticle(particle.getPrevParticle(), 
//                particle.getFilter(), 
//                particle.getObs(), 
//                particle.getState(), 
//                stateSample);

        final double transStateLogLik =
            this.updater.computeLogLikelihood(transStateTmp, data)
                + particlePriorLogLik;

        final GaussianEnsParticle transState =
            this.forwardParticle(particle, data);

        logLikelihoods.add(transStateLogLik);
        particleSupport.add(transState);

        particleTotalLogLikelihood =
            ExtLogMath.add(particleTotalLogLikelihood, transStateLogLik
                + Math.log(particleCount));
      }
    }

    final CountedDataDistribution<GaussianEnsParticle> resampledParticles;
    /*
     * Water-filling resample, for a smoothed predictive set
     */
    resampledParticles =
        ExtSamplingUtils.waterFillingResample(
            Doubles.toArray(logLikelihoods),
            particleTotalLogLikelihood, particleSupport,
            this.random, this.numParticles);
    ResampleType resampleType = ((WFCountedDataDistribution) resampledParticles)
          .wasWaterFillingApplied() ?
            ResampleType.WATER_FILLING:
              ResampleType.NO_REPLACEMENT;

    /*
     * Update sufficient stats. 
     */
    final CountedDataDistribution<GaussianEnsParticle> updatedDist =
        new CountedDataDistribution<GaussianEnsParticle>(true);
    for (final Entry<GaussianEnsParticle, MutableDouble> entry : resampledParticles
        .asMap().entrySet()) {
      final GaussianEnsParticle updatedEntry =
          this.updater.update(entry.getKey());
      updatedEntry.setResampleType(resampleType);
      updatedEntry.setStateLogWeight(entry.getValue().doubleValue());
      updatedDist.set(updatedEntry, entry.getValue().doubleValue(),
          ((MutableDoubleCount) entry.getValue()).count);
    }

    Preconditions
        .checkState(updatedDist.getTotalCount() == this.numParticles);
    target.clear();
    target.incrementAll(updatedDist);
    Preconditions.checkState(((CountedDataDistribution) target)
        .getTotalCount() == this.numParticles);
  }
}
