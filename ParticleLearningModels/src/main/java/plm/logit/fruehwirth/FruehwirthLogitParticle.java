package plm.logit.fruehwirth;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;

public class FruehwirthLogitParticle extends AbstractCloneableSerializable {

  protected UnivariateGaussian EVcomponent;
  protected FruehwirthLogitParticle previousParticle;
  protected KalmanFilter regressionFilter;
  protected MultivariateGaussian linearState;
  protected Vector augResponseSample;
  protected double priorPredMean;
  protected double priorPredCov;

  public Vector getAugResponseSample() {
    return this.augResponseSample;
  }

  public void setAugResponseSample(Vector augResponseSample) {
    this.augResponseSample = augResponseSample;
  }

  public FruehwirthLogitParticle(
      FruehwirthLogitParticle previousParticle, KalmanFilter linearComponent,
      MultivariateGaussian linearState, UnivariateGaussian EVcomponent) {
    this.previousParticle = previousParticle;
    this.regressionFilter = linearComponent;
    this.linearState = linearState;
    this.EVcomponent = EVcomponent;
  }

  @Override
  public FruehwirthLogitParticle clone() {
    FruehwirthLogitParticle clone = (FruehwirthLogitParticle) super.clone();
    clone.EVcomponent = this.EVcomponent;
    clone.previousParticle = this.previousParticle;
    clone.regressionFilter = this.regressionFilter.clone();
    clone.linearState = this.linearState.clone();
    clone.augResponseSample = this.augResponseSample;
    clone.priorPredMean = this.priorPredMean;
    clone.priorPredCov = this.priorPredCov;
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

  public KalmanFilter getRegressionFilter() {
    return regressionFilter;
  }

  public void setRegressionFilter(KalmanFilter linearComponent) {
    this.regressionFilter = linearComponent;
  }

  public void setPriorPredMean(double predPriorObsMean) {
    this.priorPredMean = predPriorObsMean;
  }

  public void setPriorPredCov(double predPriorObsCov) {
    this.priorPredCov = predPriorObsCov;
  }

  public double getPriorPredMean() {
    return this.priorPredMean;
  }

  public double getPriorPredCov() {
    return this.priorPredCov;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("FruehwirthLogitParticle [linearState=")
        .append(this.linearState).append(", augResponseSample=")
        .append(this.augResponseSample).append(", predPriorObsMean=")
        .append(this.priorPredMean).append(", predPriorObsCov=")
        .append(this.priorPredCov).append("]");
    return builder.toString();
  }
  
  
}
