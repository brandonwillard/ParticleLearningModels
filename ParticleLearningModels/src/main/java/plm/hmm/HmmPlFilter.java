package plm.hmm;

import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.AbstractParticleFilter;
import gov.sandia.cognition.util.AbstractCloneableSerializable;

import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import plm.hmm.HmmTransitionState.ResampleType;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.ExtLogMath;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.statistics.ExtSamplingUtils;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;
import com.statslibextensions.statistics.distribution.WFCountedDataDistribution;
import com.statslibextensions.util.ObservedValue;

public abstract class HmmPlFilter<HmmType extends GenericHMM<ResponseType, ?, ?>, ParticleType extends HmmTransitionState<ResponseType, HmmType>, ResponseType>
    extends
    AbstractParticleFilter<ObservedValue<ResponseType, Void>, ParticleType> {

  public static abstract class HmmPlUpdater<HmmType extends GenericHMM<ResponseType, ?, ?>, ParticleType extends HmmTransitionState<ResponseType, HmmType>, ResponseType> extends
      AbstractCloneableSerializable implements
      Updater<ObservedValue<ResponseType, Void>, ParticleType> {

    private static final long serialVersionUID = 1675005722404209890L;

    final protected HmmType priorHmm;
    final protected Random rng;

    public HmmPlUpdater(HmmType hmm, Random rng) {
      this.priorHmm = hmm;
      this.rng = rng;
    }

    public abstract WFCountedDataDistribution<ParticleType>
        baumWelchInitialization(List<ResponseType> sample,
          final int numParticles);
    
  }

  private static final long serialVersionUID = 2271089378484039661L;

  protected boolean resampleOnly;

  public HmmPlFilter(boolean resampleOnly) {
    super();
    this.resampleOnly = resampleOnly;
  }

  public boolean isResampleOnly() {
    return resampleOnly;
  }

  public void setResampleOnly(boolean resampleOnly) {
    this.resampleOnly = resampleOnly;
  }

  @Override
  public void update(
    DataDistribution<ParticleType> target,
    ObservedValue<ResponseType, Void> data) {

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final List<Double> logLikelihoods = Lists.newArrayList();
    final List<ParticleType> particleSupport =
        Lists.newArrayList();
    for (final ParticleType particle : target
        .getDomain()) {
      final HmmType hmm = particle.getHmm();

      final int particleCount =
          ((CountedDataDistribution) target).getCount(particle);

      final double particlePriorLogLik =
          target.getLogFraction(particle);

      for (int i = 0; i < particle.getHmm().getNumStates(); i++) {

        final ParticleType transState =
            this.propagate(particle, i, data);

        final double transStateLogLik =
            this.updater.computeLogLikelihood(transState, data)
                + particlePriorLogLik
                + Math.log(hmm.getTransitionProbability().getElement(
                    i, particle.getClassId()));

        logLikelihoods.add(transStateLogLik);
        particleSupport.add(transState);
        if (particleCount - 1 > 0) {
          logLikelihoods.addAll(Collections.nCopies(
              particleCount - 1, transStateLogLik));
          // FIXME this casting is no good.
          particleSupport.addAll(Collections.nCopies(
              particleCount - 1, (ParticleType)transState.clone()));
        }

        particleTotalLogLikelihood =
            ExtLogMath.add(particleTotalLogLikelihood, transStateLogLik
                + Math.log(particleCount));
      }
    }

    final ResampleType resampleType;
    final CountedDataDistribution<ParticleType> resampledParticles;
    if (this.resampleOnly) {
      resampledParticles = new CountedDataDistribution<ParticleType>(true);
      resampledParticles.incrementAll(ExtSamplingUtils
          .sampleReplaceCumulativeLogScale(
              ExtSamplingUtils.accumulate(logLikelihoods),
              particleSupport,
              this.random, this.numParticles));
      resampleType = ResampleType.REPLACEMENT;
    } else {
      /*
       * Water-filling resample, for a smoothed predictive set
       */
      resampledParticles =
          ExtSamplingUtils.waterFillingResample(
              Doubles.toArray(logLikelihoods),
              particleTotalLogLikelihood, particleSupport,
              this.random, this.numParticles);
      resampleType = 
          ((WFCountedDataDistribution) resampledParticles)
              .wasWaterFillingApplied() ?
                  ResampleType.WATER_FILLING:
                    ResampleType.NO_REPLACEMENT;
    }

    /*
     * Propagate
     */
    final CountedDataDistribution<ParticleType> updatedDist =
        new CountedDataDistribution<ParticleType>(true);
    for (final Entry<ParticleType, MutableDouble> entry : resampledParticles
        .asMap().entrySet()) {
      final ParticleType updatedEntry =
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

  protected abstract ParticleType propagate(
      ParticleType particle, int i, ObservedValue<ResponseType, Void> data);

}