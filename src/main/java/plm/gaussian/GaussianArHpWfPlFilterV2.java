package plm.gaussian;

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
import gov.sandia.cognition.util.AbstractCloneableSerializable;

import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.log4j.Logger;

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
 * A particle filter for a multivariate Gaussian AR(1) obs model
 * with shared state and obs covariance hyper priors (via parameter learning).
 * Each step for a particle samples AR(1) components multiple times; all combined
 * particle samples are then resampled via water-filled. 
 * 
 * This version samples intermediate sigma2 values.
 * 
 * @author bwillard
 * 
 */
public class GaussianArHpWfPlFilterV2 extends AbstractParticleFilter<ObservedValue<Vector, ?>, GaussianArHpWfParticle> {
  public class GaussianArHpWfPlUpdater extends AbstractCloneableSerializable implements
      Updater<ObservedValue<Vector, ?>, GaussianArHpWfParticle> {


    /*
     * Prior scale mixure for system and measurement equations.
     */
    final protected InverseGammaDistribution initialPriorSigma2;

    /*
     * Prior system const. and AR(1) terms as a stacked vector, in that order.
     */
    final protected MultivariateGaussian initialPriorPsi;

    final protected KalmanFilter initialKf;
    final protected Random rng;
    
    public GaussianArHpWfPlUpdater(KalmanFilter kf, 
        InverseGammaDistribution priorSigma2, 
        MultivariateGaussian priorPsi, Random rng) {
      this.rng = rng;
      this.initialKf = kf;
      this.initialPriorSigma2 = priorSigma2;
      this.initialPriorPsi = priorPsi;
    }

    @Override
    public double computeLogLikelihood(
      GaussianArHpWfParticle transState,
      ObservedValue<Vector,?> observation) {

      final MultivariateGaussian priorPredState = transState.getState();
      final KalmanFilter kf = transState.getFilter();
      /*
       * Construct the measurement prior predictive likelihood
       */
      final Vector mPriorPredMean = kf.getModel().getC().times(priorPredState.getMean());
      final Matrix mPriorPredCov = kf.getModel().getC().times(priorPredState.getCovariance())
          .times(kf.getModel().getC().transpose())
          .plus(kf.getMeasurementCovariance());
      final MultivariateGaussian mPriorPredDist = new MultivariateGaussian(
          mPriorPredMean, mPriorPredCov); 

      final double logCt = mPriorPredDist.getProbabilityFunction().logEvaluate(
          observation.getObservedValue());

      return logCt;
    }

    @Override
    public DataDistribution<GaussianArHpWfParticle> createInitialParticles(
        int numParticles) {
      final CountedDataDistribution<GaussianArHpWfParticle> initialParticles =
          CountedDataDistribution.create(numParticles, true);
      for (int i = 0; i < numParticles; i++) {

        final InverseGammaDistribution thisPriorScale = this.initialPriorSigma2.clone();

        final KalmanFilter thisKf = this.initialKf.clone();
        /*
         * In this model, covariance is the same across components;
         * the constant offset varies.
         * As well, we need to set/reset the kalman filters to adhere
         * to the intended model.
         */
        final double scaleSample = thisPriorScale.sample(this.rng);

        final MultivariateGaussian thisPriorOffset = initialPriorPsi.clone();

        final Vector systemSample = thisPriorOffset.sample(this.rng);
        final Vector offsetTerm = systemSample.subVector(0, 
            systemSample.getDimensionality()/2 - 1);
        thisKf.getModel().setState(offsetTerm);
        thisKf.setCurrentInput(offsetTerm);

        final Matrix A = MatrixFactory.getDefault().createDiagonal(
            systemSample.subVector(
                systemSample.getDimensionality()/2, 
                systemSample.getDimensionality() - 1));
        thisKf.getModel().setA(A);

        final Matrix offsetIdent = MatrixFactory.getDefault().createIdentity(
            systemSample.getDimensionality()/2, systemSample.getDimensionality()/2);
        thisKf.getModel().setB(offsetIdent);

        final Matrix measIdent = MatrixFactory.getDefault().createIdentity(
            thisKf.getModel().getOutputDimensionality(), 
            thisKf.getModel().getOutputDimensionality());
        thisKf.setMeasurementCovariance(measIdent.scale(scaleSample));

        final Matrix modelIdent = MatrixFactory.getDefault().createIdentity(
            thisKf.getModel().getStateDimensionality(), 
            thisKf.getModel().getStateDimensionality());
        thisKf.setModelCovariance(modelIdent.scale(scaleSample));

        final MultivariateGaussian priorState = thisKf.createInitialLearnedObject();
        final Vector priorStateSample = priorState.sample(this.rng);

        final GaussianArHpWfParticle particle =
            new GaussianArHpWfParticle(thisKf, 
                ObservedValue.<Vector>create(0, null), priorState, 
                priorStateSample,
                thisPriorScale, thisPriorOffset,
                scaleSample, null);
        particle.setResampleType(ResampleType.NONE);

        final double logWeight = -Math.log(numParticles);
        particle.setStateLogWeight(logWeight);
        initialParticles.increment(particle, logWeight);
      }
      return initialParticles;
    }

    @Override
    public GaussianArHpWfParticle update(
      GaussianArHpWfParticle predState) {

      final MultivariateGaussian posteriorState = predState.getState().clone();
      final KalmanFilter kf = predState.getFilter().clone();
      kf.update(posteriorState, predState.getObservation().getObservedValue());

      /*
       * The following are the parameter learning updates;
       * they can be done off-line, but we'll do them now.
       * TODO FIXME check that the input/offset thing is working!
       */
      final InverseGammaDistribution scaleSS = predState.getSigma2SS().clone();
      final MultivariateGaussian systemOffsetsSS = predState.getPsiSS().clone();

      final int xDim = posteriorState.getInputDimensionality();
      final Matrix Ij = MatrixFactory.getDefault().createIdentity(xDim, xDim);
      final Matrix H = MatrixFactory.getDefault().createMatrix(xDim, xDim * 2);
      H.setSubMatrix(0, 0, Ij);
      H.setSubMatrix(0, xDim, MatrixFactory.getDefault().createDiagonal(predState.getStateSample()));
      final Vector postStateSample = posteriorState.sample(this.rng);
      final MultivariateGaussian priorPhi = predState.getPsiSS();
      final Vector phiPriorSmpl = priorPhi.sample(this.rng);
      final Vector xHdiff = postStateSample.minus(H.times(phiPriorSmpl));

      final double newN = scaleSS.getShape() + 1d;
      final double d = scaleSS.getScale() + xHdiff.dotProduct(xHdiff);
      
      scaleSS.setScale(d);
      scaleSS.setShape(newN);
      
      // FIXME TODO: crappy sampler
      final double newScaleSmpl = scaleSS.sample(this.rng);
      
      /*
       * Update state and measurement covariances, which
       * have a strict dependency in this model (equality).
       */
      kf.setMeasurementCovariance(MatrixFactory.getDefault().createDiagonal(
          VectorFactory.getDefault().createVector(kf.getModel().getOutputDimensionality(), 
              newScaleSmpl)));

      kf.setModelCovariance(MatrixFactory.getDefault().createDiagonal(
          VectorFactory.getDefault().createVector(kf.getModel().getStateDimensionality(), 
              newScaleSmpl)));

      /*
       * Update offset and AR(1) prior(s).
       * Note that we divide out the previous scale param, since
       * we want to update A alone.
       */
      final Matrix priorAInv = priorPhi.getCovariance().scale(1d/predState.getSigma2Sample()).inverse();
      /*
       * TODO FIXME: we don't have a generalized outer product, so we're only
       * supporting the 1d case for now.
       */
      final Vector Hv = H.convertToVector();
      final Matrix postAInv = priorAInv.plus(Hv.outerProduct(Hv)).inverse();
      // TODO FIXME: ewww.  inverse.
      final Vector postPhiMean = postAInv.times(priorAInv.times(phiPriorSmpl).plus(
          H.transpose().times(postStateSample)));
      final MultivariateGaussian postPhi = systemOffsetsSS;
      postPhi.setMean(postPhiMean);
      postPhi.setCovariance(postAInv.scale(newScaleSmpl));
      
      final Vector postPhiSmpl = postPhi.sample(this.rng);
      final Matrix smplArTerms = MatrixFactory.getDefault().createDiagonal(
          postPhiSmpl.subVector(
              postPhiSmpl.getDimensionality()/2, 
              postPhiSmpl.getDimensionality() - 1));
      kf.getModel().setA(smplArTerms);

      final Vector smplOffsetTerm = postPhiSmpl.subVector(0, 
              postPhiSmpl.getDimensionality()/2 - 1);
      kf.getModel().setState(smplOffsetTerm);
      kf.setCurrentInput(smplOffsetTerm);
  
      final GaussianArHpWfParticle postState =
          new GaussianArHpWfParticle(kf, predState.getObservation(), 
              posteriorState, postStateSample, scaleSS, systemOffsetsSS, newScaleSmpl, null);

      return postState;
    }

  }

  final Logger log = Logger.getLogger(GaussianArHpWfPlFilterV2.class);
  final int numSubSamples;

  public GaussianArHpWfPlFilterV2(KalmanFilter kf, 
      InverseGammaDistribution priorScale, MultivariateGaussian priorSysOffset,
      Random rng, int numSubSamples, boolean resampleOnly) {
    this.numSubSamples = numSubSamples;
    this.setUpdater(new GaussianArHpWfPlUpdater(kf, priorScale, priorSysOffset, rng));
    this.setRandom(rng);
  }

  protected GaussianArHpWfParticle propagate(
      GaussianArHpWfParticle prevState, ObservedValue<Vector,?> data) {
    /*
     * Perform the filtering step
     */
    MultivariateGaussian priorPredictedState = prevState.getState().clone(); 
    final KalmanFilter kf = prevState.getFilter().clone();
    kf.predict(priorPredictedState);
    
    final InverseGammaDistribution scaleSS = prevState.getSigma2SS().clone();
    final MultivariateGaussian systemSS = prevState.getPsiSS().clone();

    final GaussianArHpWfParticle newTransState =
        new GaussianArHpWfParticle(prevState, kf,
            data, priorPredictedState, prevState.getStateSample(), 
            scaleSS, systemSS, prevState.getSigma2Sample(), null);

    return newTransState;
  }

  /**
   * This update will split each particle off into K sub-samples, then water-fill.
   */
  @Override
  public void update(
    DataDistribution<GaussianArHpWfParticle> target,
    ObservedValue<Vector, ?> data) {

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final List<Double> logLikelihoods = Lists.newArrayList();
    final List<GaussianArHpWfParticle> particleSupport =
        Lists.newArrayList();
    for (final GaussianArHpWfParticle particle : target
        .getDomain()) {

      final int particleCount =
          ((CountedDataDistribution) target).getCount(particle);

      final double particlePriorLogLik =
          target.getLogFraction(particle);

      for (int i = 0; i < this.numSubSamples * particleCount; i++) {

        final GaussianArHpWfParticle transState =
            this.propagate(particle, data);

        final double transStateLogLik =
            this.updater.computeLogLikelihood(transState, data)
                + particlePriorLogLik;

        logLikelihoods.add(transStateLogLik);
        particleSupport.add(transState);

        particleTotalLogLikelihood =
            ExtLogMath.add(particleTotalLogLikelihood, transStateLogLik
                + Math.log(particleCount));
      }
    }

    final CountedDataDistribution<GaussianArHpWfParticle> resampledParticles;
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
     * Propagate
     */
    final CountedDataDistribution<GaussianArHpWfParticle> updatedDist =
        new CountedDataDistribution<GaussianArHpWfParticle>(true);
    for (final Entry<GaussianArHpWfParticle, MutableDouble> entry : resampledParticles
        .asMap().entrySet()) {
      final GaussianArHpWfParticle updatedEntry =
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
