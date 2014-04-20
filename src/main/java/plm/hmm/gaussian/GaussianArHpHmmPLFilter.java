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
 * where the mixture components follow a HMM. I.e.
 * \[
 *  y_t = F_t x_t + v_t, \, v_t \sim N(0, V_t \phi)  \\
 *  x_t = \alpha + \beta x_{t-1} + w_t, \, w_t \sim N(0, W_t \phi) 
 * \] 
 * where \(\psi = [\alpha, \beta] \sim N(m^\psi, C^\psi)\),  \(\phi \sim IG(n, S)\) (the inverse scale), 
 * \(x_t \sim N(m_t^x, C_t^x)\) (the state).
 * 
 * @author Brandon Willard
 * 
 */
public class GaussianArHpHmmPLFilter extends HmmPlFilter<DlmHiddenMarkovModel, GaussianArHpTransitionState, Vector> {

  public class GaussianArHpHmmPlUpdater extends HmmPlUpdater<DlmHiddenMarkovModel, GaussianArHpTransitionState, Vector> {

    /*
     * Prior scale mixure for system and measurement equations.
     */
    final protected InverseGammaDistribution priorInvScale;

    /*
     * Prior system const. term offset and AR(1) as a stacked vector, respectively.
     */
    final protected List<MultivariateGaussian> priorOffsets;
    
    public GaussianArHpHmmPlUpdater(DlmHiddenMarkovModel priorHmm, 
        InverseGammaDistribution priorInvScale, 
        List<MultivariateGaussian> priorOffsets, Random rng) {
      super(priorHmm, rng);
      this.priorInvScale = priorInvScale;
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

        final InverseGammaDistribution thisPriorInvScale = this.priorInvScale.clone();

        final DlmHiddenMarkovModel particlePriorHmm = this.priorHmm.clone();
        /*
         * In this model, covariance is the same across components;
         * the constant offset varies.
         * As well, we need to set/reset the kalman filters to adhere
         * to the intended model.
         */
        final List<MultivariateGaussian> thesePriorOffsets = Lists.newArrayList();
        final double invScaleSample = thisPriorInvScale.sample(this.rng);
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
          kf.setMeasurementCovariance(measIdent.scale(invScaleSample));

          final Matrix modelIdent = MatrixFactory.getDefault().createIdentity(
              kf.getModel().getStateDimensionality(), 
              kf.getModel().getStateDimensionality());
          kf.setModelCovariance(modelIdent.scale(invScaleSample));
        }

        final KalmanFilter kf = Iterables.get(particlePriorHmm.getStateFilters(), 
            sampledClass);
        final MultivariateGaussian priorState = kf.createInitialLearnedObject();
        final Vector priorStateSample = priorState.sample(this.rng);

        final GaussianArHpTransitionState particle =
            new GaussianArHpTransitionState(particlePriorHmm, sampledClass,
                ObservedValue.<Vector>create(0, null), priorState, 
                priorStateSample,
                thisPriorInvScale, thesePriorOffsets,
                invScaleSample);

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
      final InverseGammaDistribution invScaleSS = predState.getInvScaleSS().clone();
      final List<MultivariateGaussian> systemOffsetsSS =
          ObjectUtil.cloneSmartElementsAsArrayList(predState.getPsiSS());

      final int xDim = posteriorState.getInputDimensionality();
      final Matrix Ij = MatrixFactory.getDefault().createIdentity(xDim, xDim);
      final Matrix H = MatrixFactory.getDefault().createMatrix(xDim, xDim * 2);
      H.setSubMatrix(0, 0, Ij);
      H.setSubMatrix(0, xDim, MatrixFactory.getDefault().createDiagonal(predState.getStateSample()));
      final Vector postStateSample = posteriorState.sample(this.rng);
      final MultivariateGaussian priorPhi = predState.getPsiSS().get(predState.getClassId());
      final Vector phiPriorSmpl = priorPhi.sample(this.rng);
      final Vector xHdiff = postStateSample.minus(H.times(phiPriorSmpl));

      final double newN = invScaleSS.getShape() + 1d;
      final double d = invScaleSS.getScale() + xHdiff.dotProduct(xHdiff);
      
      invScaleSS.setScale(d);
      invScaleSS.setShape(newN);
      
      // FIXME TODO: crappy sampler
      final double newInvScaleSmpl = invScaleSS.sample(this.rng);
      
      /*
       * Update state and measurement covariances, which
       * have a strict dependency in this model (equality).
       */
      kf.setMeasurementCovariance(MatrixFactory.getDefault().createDiagonal(
          VectorFactory.getDefault().createVector(kf.getModel().getOutputDimensionality(), 
              newInvScaleSmpl)));

      kf.setModelCovariance(MatrixFactory.getDefault().createDiagonal(
          VectorFactory.getDefault().createVector(kf.getModel().getStateDimensionality(), 
              newInvScaleSmpl)));

      /*
       * Update offset and AR(1) prior(s).
       * Note that we divide out the previous inv scale param, since
       * we want to update A alone.
       */
      final Matrix priorAInv = priorPhi.getCovariance().scale(1d/predState.getInvScaleSample()).inverse();
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
      postPhi.setCovariance(postAInv.scale(newInvScaleSmpl));
      
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
              posteriorState, postStateSample, invScaleSS, systemOffsetsSS, newInvScaleSmpl);

      return postState;
    }

  }

  final Logger log = Logger.getLogger(GaussianArHpHmmPLFilter.class);

  public GaussianArHpHmmPLFilter(DlmHiddenMarkovModel hmm, 
      InverseGammaDistribution priorInvScale, List<MultivariateGaussian> priorSysOffsets,
      Random rng, boolean resampleOnly) {
    super(resampleOnly);
    this.setUpdater(new GaussianArHpHmmPlUpdater(hmm, priorInvScale, priorSysOffsets, rng));
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
    final InverseGammaDistribution invScaleSS = prevState.getInvScaleSS().clone();
    final List<MultivariateGaussian> psiSS = 
        ObjectUtil.cloneSmartElementsAsArrayList(prevState.getPsiSS());

    final GaussianArHpTransitionState newTransState =
        new GaussianArHpTransitionState(prevState, newHmm,
            predClass, data, priorPredictedState, prevState.getStateSample(), 
            invScaleSS, psiSS, prevState.getInvScaleSample());

    return newTransState;
  }

}
