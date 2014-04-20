package plm.ensemble;

import plm.hmm.HmmTransitionState.ResampleType;

import com.statslibextensions.util.ObservedValue;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;

public abstract class GaussianEnsParticle extends AbstractCloneableSerializable {

  protected double logWeight = Double.NEGATIVE_INFINITY;
  protected GaussianHpEnsParticle prevParticle;
  protected KalmanFilter kf;
  protected ObservedValue<Vector, ?> obs;
  protected MultivariateGaussian state;
  protected Vector stateSample;
  protected ResampleType resampleType;

  public GaussianEnsParticle() {
    super();
  }

  public GaussianEnsParticle(double logWeight, GaussianHpEnsParticle prevParticle, KalmanFilter kf,
    ObservedValue<Vector, ?> obs, MultivariateGaussian state, Vector stateSample, ResampleType resampleType) {
    super();
    this.logWeight = logWeight;
    this.prevParticle = prevParticle;
    this.kf = kf;
    this.obs = obs;
    this.state = state;
    this.stateSample = stateSample;
    this.resampleType = resampleType;
  }

  public ResampleType getResampleType() {
    return resampleType;
  }

  public KalmanFilter getKf() {
    return kf;
  }

  public ObservedValue<Vector, ?> getObs() {
    return obs;
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

  public Vector getStateSample() {
    return this.stateSample;
  }

  public GaussianHpEnsParticle getPrevParticle() {
    return this.prevParticle;
  }

  public void setPrevParticle(GaussianHpEnsParticle prevParticle) {
    this.prevParticle = prevParticle;
  }

  public double getLogWeight() {
    return this.logWeight;
  }

  @Override
  public GaussianEnsParticle clone() {
    GaussianEnsParticle clone = (GaussianEnsParticle) super.clone();
    clone.kf = 
        new KalmanFilter(
            new LinearDynamicalSystem(
                this.kf.getModel().getA(),
                this.kf.getModel().getB(),
                this.kf.getModel().getC()), 
            this.kf.getModelCovariance(), 
            this.kf.getMeasurementCovariance());
    clone.state = new MultivariateGaussian(
        this.state.getMean(), this.state.getCovariance());
    clone.stateSample = this.stateSample.clone();
    clone.logWeight = this.logWeight;
    clone.obs = this.obs;
    clone.prevParticle = this.prevParticle;
    return clone;
  }

  public void setResampleType(ResampleType resampleType) {
    this.resampleType = resampleType;
  }

}