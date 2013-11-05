package plm.logit.fruehwirth;

import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;

public class FruehwirthLogitParticle extends AbstractCloneableSerializable {

  protected UnivariateGaussian EVcomponent;
  protected FruehwirthLogitParticle previousParticle;
  protected KalmanFilter linearComponent;
  protected MultivariateGaussian linearState;

  public FruehwirthLogitParticle(
      FruehwirthLogitParticle previousParticle, KalmanFilter linearComponent,
      MultivariateGaussian linearState, UnivariateGaussian EVcomponent) {
    this.previousParticle = previousParticle;
    this.linearComponent = linearComponent;
    this.linearState = linearState;
    this.EVcomponent = EVcomponent;
  }

  @Override
  public FruehwirthLogitParticle clone() {
    FruehwirthLogitParticle clone = (FruehwirthLogitParticle) super.clone();
    clone.EVcomponent = this.EVcomponent;
    clone.previousParticle = this.previousParticle;
    clone.linearComponent = this.linearComponent.clone();
    clone.linearState = this.linearState.clone();
    return clone;
  }

  public UnivariateGaussian getEVcomponent() {
    return this.EVcomponent;
  }

  public void setEVcomponent(UnivariateGaussian componentDist) {
    this.EVcomponent = componentDist;
  }

  public MultivariateGaussian getLinearState() {
    return linearState;
  }

  public void setLinearState(MultivariateGaussian linearState) {
    this.linearState = linearState;
  }

  public FruehwirthLogitParticle getPreviousParticle() {
    return previousParticle;
  }

  public void setPreviousParticle(FruehwirthLogitParticle previousParticle) {
    this.previousParticle = previousParticle;
  }

  public KalmanFilter getLinearComponent() {
    return linearComponent;
  }

  public void setLinearComponent(KalmanFilter linearComponent) {
    this.linearComponent = linearComponent;
  }
  
  
}
