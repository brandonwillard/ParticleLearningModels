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
  private ObservedValue<Vector, Void> obs;
  private MultivariateGaussian state;
  private Vector stateSample;
  private InverseGammaDistribution scaleSS;
  private MultivariateGaussian systemSS;
  private double scaleSample;
  private ResampleType resampleType;

  public GaussianArHpWfParticle(KalmanFilter thisKf,
      ObservedValue<Vector, Void> create, MultivariateGaussian priorState,
      Vector priorStateSample, InverseGammaDistribution thisPriorScale,
      MultivariateGaussian thisPriorOffset, double scaleSample) {
    this(null, thisKf, create, priorState, priorStateSample, thisPriorScale, thisPriorOffset, scaleSample);
  }

  public GaussianArHpWfParticle(GaussianArHpWfParticle prevParticle,
      KalmanFilter kf, ObservedValue<Vector, Void> obs,
      MultivariateGaussian state, Vector stateSample,
      InverseGammaDistribution scaleSS, MultivariateGaussian systemSS,
      double scaleSample) {
    this.prevParticle = prevParticle;
    this.kf = kf;
    this.obs = obs;
    this.state = state;
    this.stateSample = stateSample;
    this.scaleSS = scaleSS;
    this.systemSS = systemSS;
    this.scaleSample = scaleSample;
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

  public ObservedValue<Vector, Void> getObservation() {
    return this.obs;
  }

  public InverseGammaDistribution getScaleSS() {
    return this.scaleSS;
  }

  public MultivariateGaussian getSystemOffsetSS() {
    return this.systemSS;
  }

  public Vector getStateSample() {
    return this.stateSample;
  }

  public double getScaleSample() {
    return this.scaleSample;
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
    clone.scaleSample = this.scaleSample;
    clone.scaleSS = this.scaleSS.clone();
    clone.state = this.state.clone();
    clone.stateSample = this.stateSample.clone();
    clone.systemSS = this.systemSS.clone();
    return clone;
  }

  public void setResampleType(ResampleType resampleType) {
    this.resampleType = resampleType;
  }

}
