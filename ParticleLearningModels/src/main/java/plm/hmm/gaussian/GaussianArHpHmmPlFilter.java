package plm.hmm.gaussian;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.ObjectUtil;

import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import plm.hmm.DlmHiddenMarkovModel;
import plm.hmm.HmmPlFilter;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;
import com.statslibextensions.statistics.distribution.WFCountedDataDistribution;
import com.statslibextensions.util.ObservedValue;

/**
 * A Particle Learning filter for a multivariate Gaussian AR(1) mixture obs model
 * with shared state and obs covariance hyper priors (via parameter learning) 
 * where the mixture components follow a HMM.
 * 
 * @author bwillard
 * 
 */
public class GaussianArHpHmmPlFilter extends HmmPlFilter<DlmHiddenMarkovModel, GaussianArHpTransitionState, Vector> {

  public class GaussianArHpHmmPlUpdater extends HmmPlUpdater<DlmHiddenMarkovModel, GaussianArHpTransitionState, Vector> {

    /*
     * Prior scale mixure for system and measurement equations.
     */
    final protected InverseGammaDistribution priorScale;

    /*
     * Prior system const. term offset and AR(1) as a stacked vector, respectively.
     */
    final protected List<MultivariateGaussian> priorOffsets;
    
    public GaussianArHpHmmPlUpdater(DlmHiddenMarkovModel priorHmm, 
        InverseGammaDistribution priorScale, 
        List<MultivariateGaussian> priorOffsets, Random rng) {
      super(priorHmm, rng);
      this.priorScale = priorScale;
      this.priorOffsets = priorOffsets;
    }

    @Override
    public double computeLogLikelihood(
      GaussianArHpTransitionState transState,
      ObservedValue<Vector,Void> observation) {

      final MultivariateGaussian priorPredState = transState.getState();
      final KalmanFilter kf = Iterables.get(transState.getHmm().getStateFilters(),
          transState.getClassId());
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
    public WFCountedDataDistribution<GaussianArHpTransitionState> baumWelchInitialization(
        List<Vector> sample, int numParticles) {
      Preconditions.checkState(false);
      return null;
//      new WFCountedDataDistribution<DlmTransitionState>(
//          this.createInitialParticles(numParticles), true);
    }


    @Override
    public DataDistribution<GaussianArHpTransitionState> createInitialParticles(
        int numParticles) {
      final CountedDataDistribution<GaussianArHpTransitionState> initialParticles =
          CountedDataDistribution.create(numParticles, true);
      for (int i = 0; i < numParticles; i++) {

        final int sampledClass =
            DiscreteSamplingUtil.sampleIndexFromProbabilities(
                this.rng, this.priorHmm.getClassMarginalProbabilities());

        final InverseGammaDistribution thisPriorScale = this.priorScale.clone();

        final DlmHiddenMarkovModel particlePriorHmm = this.priorHmm.clone();
        /*
         * In this model, covariance is the same across components;
         * the constant offset varies.
         * As well, we need to set/reset the kalman filters to adhere
         * to the intended model.
         */
        final List<MultivariateGaussian> thesePriorOffsets = Lists.newArrayList();
        final double scaleSample = thisPriorScale.sample(this.rng);
        int k = 0;
        for (KalmanFilter kf : particlePriorHmm.getStateFilters()) {
          final MultivariateGaussian thisPriorOffset = priorOffsets.get(k).clone();
          thesePriorOffsets.add(thisPriorOffset);
          k++;

          final Vector systemSample = thisPriorOffset.sample(this.rng);
          final Vector offsetTerm = systemSample.subVector(0, 
              systemSample.getDimensionality()/2 - 1);
          kf.getModel().setState(offsetTerm);
          kf.setCurrentInput(offsetTerm);

          final Matrix A = MatrixFactory.getDefault().createDiagonal(
              systemSample.subVector(
                  systemSample.getDimensionality()/2, 
                  systemSample.getDimensionality() - 1));
          kf.getModel().setA(A);

          final Matrix offsetIdent = MatrixFactory.getDefault().createIdentity(
              systemSample.getDimensionality()/2, systemSample.getDimensionality()/2);
          kf.getModel().setB(offsetIdent);

          final Matrix measIdent = MatrixFactory.getDefault().createIdentity(
              kf.getModel().getOutputDimensionality(), 
              kf.getModel().getOutputDimensionality());
          kf.setMeasurementCovariance(measIdent.scale(scaleSample));

          final Matrix modelIdent = MatrixFactory.getDefault().createIdentity(
              kf.getModel().getStateDimensionality(), 
              kf.getModel().getStateDimensionality());
          kf.setModelCovariance(modelIdent.scale(scaleSample));
        }

        final KalmanFilter kf = Iterables.get(particlePriorHmm.getStateFilters(), 
            sampledClass);
        final MultivariateGaussian priorState = kf.createInitialLearnedObject();
        final Vector priorStateSample = priorState.sample(this.rng);

        final GaussianArHpTransitionState particle =
            new GaussianArHpTransitionState(particlePriorHmm, sampledClass,
                ObservedValue.<Vector>create(0, null), priorState, 
                priorStateSample,
                thisPriorScale, thesePriorOffsets,
                scaleSample);

        final double logWeight = -Math.log(numParticles);
        particle.setStateLogWeight(logWeight);
        initialParticles.increment(particle, logWeight);
      }
      return initialParticles;
    }

    @Override
    public GaussianArHpTransitionState update(
      GaussianArHpTransitionState predState) {

      final MultivariateGaussian posteriorState = predState.getState().clone();
      final DlmHiddenMarkovModel newHmm = predState.getHmm().clone();
      KalmanFilter kf = Iterables.get(newHmm.getStateFilters(), 
          predState.getClassId());
      kf.update(posteriorState, predState.getObservation().getObservedValue());


      /*
       * The following are the parameter learning updates;
       * they can be done off-line, but we'll do them now.
       * TODO FIXME check that the input/offset thing is working!
       */
      final InverseGammaDistribution scaleSS = predState.getScaleSS().clone();
      final List<MultivariateGaussian> systemOffsetsSS =
          ObjectUtil.cloneSmartElementsAsArrayList(predState.getSystemOffsetsSS());

      final int xDim = posteriorState.getInputDimensionality();
      final Matrix Ij = MatrixFactory.getDefault().createIdentity(xDim, xDim);
      final Matrix H = MatrixFactory.getDefault().createMatrix(xDim, xDim * 2);
      H.setSubMatrix(0, 0, Ij);
      H.setSubMatrix(0, xDim, MatrixFactory.getDefault().createDiagonal(predState.getStateSample()));
      final Vector postStateSample = posteriorState.sample(this.rng);
      final MultivariateGaussian priorPhi = predState.getSystemOffsetsSS().get(predState.getClassId());
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
      final Matrix priorAInv = priorPhi.getCovariance().scale(1d/predState.getScaleSample()).inverse();
      /*
       * TODO FIXME: we don't have a generalized outer product, so we're only
       * supporting the 1d case for now.
       */
      final Vector Hv = H.convertToVector();
      final Matrix postAInv = priorAInv.plus(Hv.outerProduct(Hv)).inverse();
      // TODO FIXME: ewww.  inverse.
      final Vector postPhiMean = postAInv.times(priorAInv.times(phiPriorSmpl).plus(
          H.transpose().times(postStateSample)));
      final MultivariateGaussian postPhi = systemOffsetsSS.get(predState.getClassId());
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
  
      final GaussianArHpTransitionState postState =
          new GaussianArHpTransitionState(newHmm,
              predState.getClassId(), predState.getObservation(), 
              posteriorState, postStateSample, scaleSS, systemOffsetsSS, newScaleSmpl);

      return postState;
    }

  }

  final Logger log = Logger.getLogger(GaussianArHpHmmPlFilter.class);

  public GaussianArHpHmmPlFilter(DlmHiddenMarkovModel hmm, 
      InverseGammaDistribution priorScale, List<MultivariateGaussian> priorSysOffsets,
      Random rng, boolean resampleOnly) {
    super(resampleOnly);
    this.setUpdater(new GaussianArHpHmmPlUpdater(hmm, priorScale, priorSysOffsets, rng));
    this.setRandom(rng);
  }

  @Override
  protected GaussianArHpTransitionState propagate(
      GaussianArHpTransitionState prevState, int predClass, ObservedValue<Vector,Void> data) {
    /*
     * Perform the filtering step
     */
    MultivariateGaussian priorPredictedState = prevState.getState().clone(); 
    KalmanFilter kf = Iterables.get(prevState.getHmm().getStateFilters(), predClass);
    kf.predict(priorPredictedState);
    
    final DlmHiddenMarkovModel newHmm = prevState.getHmm().clone();
    final InverseGammaDistribution scaleSS = prevState.getScaleSS().clone();
    final List<MultivariateGaussian> systemSS = 
        ObjectUtil.cloneSmartElementsAsArrayList(prevState.getSystemOffsetsSS());

    final GaussianArHpTransitionState newTransState =
        new GaussianArHpTransitionState(prevState, newHmm,
            predClass, data, priorPredictedState, prevState.getStateSample(), 
            scaleSS, systemSS, prevState.getScaleSample());

    return newTransState;
  }

}
