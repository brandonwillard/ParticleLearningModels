package plm.gaussian;

import plm.hmm.HmmTransitionState.ResampleType;

import com.statslibextensions.util.ObservedValue;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.CloneableSerializable;

public class GaussianArHpWfParticle extends AbstractCloneableSerializable {

  private double logWeight = Double.NEGATIVE_INFINITY;
  private GaussianArHpWfParticle prevParticle;
  private KalmanFilter kf;
  private ObservedValue<Vector, ?> obs;
  private MultivariateGaussian state;
  private Vector stateSample;
  private InverseGammaDistribution sigma2SS;
  private MultivariateGaussian psiSS;
  private double sigma2Sample;
  private ResampleType resampleType;
  private Vector psiSample;

  public ResampleType getResampleType() {
    return resampleType;
  }

  public GaussianArHpWfParticle(KalmanFilter thisKf,
      ObservedValue<Vector, ?> create, MultivariateGaussian priorState,
      Vector priorStateSample, InverseGammaDistribution priorSigma2,
      MultivariateGaussian priorPsi, double sigma2Sample, Vector psiPriorSmpl) {
    this(null, thisKf, create, 
        priorState, priorStateSample, 
        priorSigma2, priorPsi, 
        sigma2Sample, psiPriorSmpl);
  }

  public GaussianArHpWfParticle(GaussianArHpWfParticle prevParticle,
      KalmanFilter kf, ObservedValue<Vector, ?> obs,
      MultivariateGaussian state, Vector stateSample,
      InverseGammaDistribution sigma2SS, MultivariateGaussian psiSS,
      double sigma2Sample, Vector psiPriorSmpl) {
    this.prevParticle = prevParticle;
    this.kf = kf;
    this.obs = obs;
    this.state = state;
    this.stateSample = stateSample;
    this.sigma2SS = sigma2SS;
    this.psiSS = psiSS;
    this.sigma2Sample = sigma2Sample;
    this.psiSample = psiPriorSmpl;
  }

  public KalmanFilter getKf() {
    return kf;
  }

  public ObservedValue<Vector, ?> getObs() {
    return obs;
  }

  public Vector getPsiSample() {
    return psiSample;
  }

  public KalmanFilter getFilter() {
    return this.kf;
  }

  public MultivariateGaussian getState() {
    return this.state;
  }

  public void setStateLogWeight(double logWeight) {
    this.logWeight = logWeight; 
  }

  public ObservedValue<Vector, ?> getObservation() {
    return this.obs;
  }

  public InverseGammaDistribution getSigma2SS() {
    return this.sigma2SS;
  }

  public MultivariateGaussian getPsiSS() {
    return this.psiSS;
  }

  public Vector getStateSample() {
    return this.stateSample;
  }

  public double getSigma2Sample() {
    return this.sigma2Sample;
  }

  public GaussianArHpWfParticle getPrevParticle() {
    return this.prevParticle;
  }

  public void setPrevParticle(GaussianArHpWfParticle prevParticle) {
    this.prevParticle = prevParticle;
  }

  public double getLogWeight() {
    return this.logWeight;
  }

  @Override
  public GaussianArHpWfParticle clone() {
    GaussianArHpWfParticle clone = (GaussianArHpWfParticle) super.clone();
    clone.kf = this.kf.clone();
    clone.logWeight = this.logWeight;
    clone.obs = this.obs;
    clone.prevParticle = this.prevParticle;
    clone.sigma2Sample = this.sigma2Sample;
    clone.sigma2SS = this.sigma2SS.clone();
    clone.state = this.state.clone();
    clone.stateSample = this.stateSample.clone();
    clone.psiSS = this.psiSS.clone();
    return clone;
  }

  public void setResampleType(ResampleType resampleType) {
    this.resampleType = resampleType;
  }

}
