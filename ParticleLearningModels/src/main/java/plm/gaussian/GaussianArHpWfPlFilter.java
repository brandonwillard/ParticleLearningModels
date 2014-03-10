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
import gov.sandia.cognition.statistics.distribution.MultivariateStudentTDistribution;
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
 * This version integrates out sigma2, utilizing t-distributions.
 * 
 * @author bwillard
 * 
 */
public class GaussianArHpWfPlFilter extends AbstractParticleFilter<ObservedValue<Vector, ?>, GaussianArHpWfParticle> {
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
    final protected Matrix Iy;
    final protected Matrix Ix;
    
    public GaussianArHpWfPlUpdater(KalmanFilter kf, 
        InverseGammaDistribution priorSigma2, 
        MultivariateGaussian priorPsi, Random rng) {
      this.rng = rng;
      this.initialKf = kf;
      this.initialPriorSigma2 = priorSigma2;
      this.initialPriorPsi = priorPsi;
      this.Iy = MatrixFactory.getDefault().createIdentity(
          kf.getModel().getOutputDimensionality(), 
          kf.getModel().getOutputDimensionality());
      this.Ix = MatrixFactory.getDefault().createIdentity(
          kf.getModel().getInputDimensionality(), 
          kf.getModel().getInputDimensionality());
    }

    @Override
    public double computeLogLikelihood(
      GaussianArHpWfParticle transParticle,
      ObservedValue<Vector,?> observation) {

      final MultivariateGaussian priorPsi = transParticle.getPsiSS();
      final KalmanFilter kf = transParticle.getFilter();

      final int xDim = kf.getModel().getInputDimensionality();
      final Matrix H = MatrixFactory.getDefault().createMatrix(xDim, xDim * 2);
      H.setSubMatrix(0, 0, Ix);
      H.setSubMatrix(0, xDim, 
          // x_{t-1}
          MatrixFactory.getDefault().createDiagonal(transParticle.getStateSample()));

      /*
       * Construct the measurement prior predictive likelihood
       * t_n (H*m^psi, d*C^psi/n) 
       */
      final Vector mPriorPredMean = H.times(priorPsi.getMean());
      final Matrix mPriorPredCov = H.times(priorPsi.getCovariance()).times(H.transpose())
          .plus(Iy.scale(2d));

      // TODO FIXME
      final Matrix stPriorPredPrec = mPriorPredCov.inverse().scale(
          transParticle.getSigma2SS().getShape()/transParticle.getSigma2SS().getScale());


      MultivariateStudentTDistribution mPriorPredDist = new MultivariateStudentTDistribution(
          transParticle.getSigma2SS().getShape(), 
          mPriorPredMean, stPriorPredPrec);

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

        final InverseGammaDistribution thisSigma2Prior = this.initialPriorSigma2.clone();
        final double sigma2Sample = thisSigma2Prior.sample(this.rng);

        final KalmanFilter thisKf = this.initialKf.clone();
        final MultivariateGaussian thisPsiPrior = initialPriorPsi.clone();
        // TODO FIXME use t-distribution
        final MultivariateGaussian thisPsiPriorSmpler = thisPsiPrior.clone();
        thisPsiPriorSmpler.getCovariance().scaleEquals(sigma2Sample);
        final Vector psiSample = thisPsiPriorSmpler.sample(this.rng);

        final Vector alphaTerm = psiSample.subVector(0, 
            psiSample.getDimensionality()/2 - 1);
        thisKf.getModel().setState(alphaTerm);
        thisKf.setCurrentInput(alphaTerm);

        final Matrix betaTerm = MatrixFactory.getDefault().createDiagonal(
            psiSample.subVector(
                psiSample.getDimensionality()/2, 
                psiSample.getDimensionality() - 1));
        thisKf.getModel().setA(betaTerm);

        final Matrix offsetIdent = MatrixFactory.getDefault().createIdentity(
            psiSample.getDimensionality()/2, psiSample.getDimensionality()/2);
        thisKf.getModel().setB(offsetIdent);

        final Matrix measIdent = Iy.clone();
        thisKf.setMeasurementCovariance(measIdent);

        final Matrix modelIdent = Ix.clone();
        thisKf.setModelCovariance(modelIdent);

        final MultivariateGaussian priorState = thisKf.createInitialLearnedObject();
        final MultivariateGaussian priorStateSmpler = thisKf.createInitialLearnedObject();
        priorStateSmpler.getCovariance().scaleEquals(sigma2Sample);
        final Vector priorStateSample = priorStateSmpler.sample(this.rng);


        final GaussianArHpWfParticle particle =
            new GaussianArHpWfParticle(thisKf, 
                ObservedValue.<Vector>create(0, null), priorState, 
                priorStateSample,
                thisSigma2Prior, thisPsiPrior,
                sigma2Sample, psiSample);
        
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

      /*
       * The following are the parameter learning updates;
       * they can be done off-line, but we'll do them now.
       * TODO FIXME check that the input/offset thing is working!
       */

      final int xDim = posteriorState.getInputDimensionality();
      final Matrix H = MatrixFactory.getDefault().createMatrix(xDim, xDim * 2);
      H.setSubMatrix(0, 0, Ix);
      H.setSubMatrix(0, xDim, 
          // x_{t-1}
          MatrixFactory.getDefault().createDiagonal(predState.getStateSample()));

      final InverseGammaDistribution sigma2SS = predState.getSigma2SS().clone();
      // TODO FIXME matrix inverse!!
      final Matrix postStatePrec = posteriorState.getCovarianceInverse().scale(
          sigma2SS.getShape()/sigma2SS.getScale());
      MultivariateStudentTDistribution postStateMarginal = new MultivariateStudentTDistribution(
          sigma2SS.getShape(), 
          posteriorState.getMean(), postStatePrec);
      final Vector postStateSample = postStateMarginal.sample(this.rng);
      
      final Vector psiPriorSmpl = predState.getPsiSample(); 
      // x_t
      final Vector xHdiff = postStateSample.minus(H.times(psiPriorSmpl));

      /*
       * 1. Update the sigma2 sufficient stats.
       */
      final double newN = sigma2SS.getShape() + 1d;
      final double d = sigma2SS.getScale() + xHdiff.dotProduct(xHdiff);
      sigma2SS.setScale(d);
      sigma2SS.setShape(newN);
      
      /*
       * 2. Update psi sufficient stats. (i.e. offset and AR(1)).
       * 
       * Note that we divide out the previous scale param, since
       * we want to update A alone.
       * TODO FIXME inverse!  ewww.
       */
      final Matrix priorAInv = predState.getPsiSS().getCovarianceInverse();
      /*
       * TODO FIXME: we don't have a generalized outer product, so we're only
       * supporting the 1d case for now.
       */
      final Vector Hv = H.convertToVector();
      /*
       * TODO FIXME inverse!  ewww.
       */
      final Matrix postAInv = priorAInv.plus(Hv.outerProduct(Hv)).inverse();
      final Vector postPsiMean = postAInv.times(priorAInv.times(psiPriorSmpl).plus(
          H.transpose().times(postStateSample)));
      final MultivariateGaussian postPsi = predState.getPsiSS().clone();
      postPsi.setMean(postPsiMean);
      postPsi.setCovariance(postAInv);
      
      final double sigma2Smpl = sigma2SS.sample(this.rng);
      final GaussianArHpWfParticle postState =
          new GaussianArHpWfParticle(kf, predState.getObservation(), 
              posteriorState, postStateSample, 
              sigma2SS, postPsi, 
              sigma2Smpl, predState.getPsiSample());

      return postState;
    }

  }

  final Logger log = Logger.getLogger(GaussianArHpWfPlFilter.class);
  final int numSubSamples;
  final protected Matrix Iy;
  final protected Matrix Ix;

  public GaussianArHpWfPlFilter(KalmanFilter kf, 
      InverseGammaDistribution priorScale, MultivariateGaussian priorSysOffset,
      Random rng, int numSubSamples, boolean resampleOnly) {
    this.numSubSamples = numSubSamples;
    this.setUpdater(new GaussianArHpWfPlUpdater(kf, priorScale, priorSysOffset, rng));
    this.setRandom(rng);
    this.Iy = MatrixFactory.getDefault().createIdentity(
        kf.getModel().getOutputDimensionality(), 
        kf.getModel().getOutputDimensionality());
    this.Ix = MatrixFactory.getDefault().createIdentity(
        kf.getModel().getInputDimensionality(), 
        kf.getModel().getInputDimensionality());
  }

  protected GaussianArHpWfParticle propagate(
      GaussianArHpWfParticle prevState, ObservedValue<Vector,?> data) {
    /*
     * Sample psi
     */
    final MultivariateGaussian priorPsi = prevState.getPsiSS();
    final Vector priorPsiSmpl = priorPsi.sample(this.getRandom());

    final KalmanFilter kf = prevState.getFilter().clone();

    /*
     * Update the filter parameters with the new psi.
     */
    final Matrix smplArTerms = MatrixFactory.getDefault().createDiagonal(
        priorPsiSmpl.subVector(
            priorPsiSmpl.getDimensionality()/2, 
            priorPsiSmpl.getDimensionality() - 1));
    kf.getModel().setA(smplArTerms);

    final Vector smplOffsetTerm = priorPsiSmpl.subVector(0, 
            priorPsiSmpl.getDimensionality()/2 - 1);
    kf.getModel().setState(smplOffsetTerm);
    kf.setCurrentInput(smplOffsetTerm);
  
    /*
     * Perform the Kalman update to get the posterior state suff. stats.
     */
    final InverseGammaDistribution priorSigma2 = prevState.getSigma2SS().clone();
    final double sigma2Sample = priorSigma2.sample(this.getRandom()); 
    MultivariateGaussian posteriorState = prevState.getState().clone(); 
    // TODO FIXME gross hack!
//    posteriorState.getCovariance().scaleEquals(sigma2Sample);
//    kf.setMeasurementCovariance(Iy.scale(sigma2Sample));
//    kf.setModelCovariance(Ix.scale(sigma2Sample));
    kf.predict(posteriorState);
    kf.update(posteriorState, data.getObservedValue());
//    kf.setMeasurementCovariance(Iy);
//    kf.setModelCovariance(Ix);
    

    final GaussianArHpWfParticle newTransState =
        new GaussianArHpWfParticle(prevState, kf,
            data, posteriorState, prevState.getStateSample(), 
            priorSigma2, priorPsi, sigma2Sample, priorPsiSmpl);

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
     * Propagate and compute prior predictive log likelihoods.
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

        /*
         * K many sub-samples of x_{t-1}  
         */
        final InverseGammaDistribution sigma2SS = particle.getSigma2SS();
        // TODO FIXME matrix inverse!!
        final Matrix postStatePrec = particle.getState().getCovarianceInverse().scale(
            sigma2SS.getShape()/sigma2SS.getScale());
        MultivariateStudentTDistribution postStateMarginal = new MultivariateStudentTDistribution(
            sigma2SS.getShape(), 
            particle.getState().getMean(), postStatePrec);
        final Vector stateSample = postStateMarginal.sample(this.getRandom());

        final GaussianArHpWfParticle transStateTmp = 
            new GaussianArHpWfParticle(particle.getPrevParticle(), 
                particle.getFilter(), 
                particle.getObs(), 
                particle.getState(), 
                stateSample, 
                sigma2SS, 
                particle.getPsiSS(), 
                particle.getSigma2Sample(), 
                particle.getPsiSample());

        final double transStateLogLik =
            this.updater.computeLogLikelihood(transStateTmp, data)
                + particlePriorLogLik;

        final GaussianArHpWfParticle transState =
            this.propagate(particle, data);

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
     * Update sufficient stats. 
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
