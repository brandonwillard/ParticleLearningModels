package plm.logit.fruehwirth.multi;

import java.util.List;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.ObjectUtil;

public class FruehwirthMultiParticle extends AbstractCloneableSerializable {

  protected UnivariateGaussian EVcomponent;
  protected FruehwirthMultiParticle previousParticle;
  protected List<KalmanFilter> regressionFilters;
  protected MultivariateGaussian linearState;
  protected Vector augResponseSample;
  protected double priorPredMean;
  protected double priorPredCov;
  protected int categoryId;

  public Vector getAugResponseSample() {
    return this.augResponseSample;
  }

  public void setAugResponseSample(Vector augResponseSample) {
    this.augResponseSample = augResponseSample;
  }

  public FruehwirthMultiParticle(
      FruehwirthMultiParticle previousParticle, KalmanFilter linearComponent,
      MultivariateGaussian linearState, UnivariateGaussian EVcomponent, int categoryId) {
    this.previousParticle = previousParticle;
    this.regressionFilters.set(categoryId, linearComponent);
    this.linearState = linearState;
    this.EVcomponent = EVcomponent;
    this.categoryId = categoryId;
  }

  @Override
  public FruehwirthMultiParticle clone() {
    FruehwirthMultiParticle clone = (FruehwirthMultiParticle) super.clone();
    clone.EVcomponent = this.EVcomponent;
    clone.previousParticle = this.previousParticle;
    clone.regressionFilters = ObjectUtil.cloneSmartElementsAsArrayList(this.regressionFilters);
    clone.linearState = this.linearState.clone();
    clone.augResponseSample = this.augResponseSample;
    clone.priorPredMean = this.priorPredMean;
    clone.priorPredCov = this.priorPredCov;
    clone.categoryId = this.categoryId;
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

  public FruehwirthMultiParticle getPreviousParticle() {
    return previousParticle;
  }

  public void setPreviousParticle(FruehwirthMultiParticle previousParticle) {
    this.previousParticle = previousParticle;
  }

  public KalmanFilter getRegressionFilter(int k) {
    return regressionFilters.get(k);
  }

  public void setRegressionFilter(int k, KalmanFilter linearComponent) {
    this.regressionFilters.set(k, linearComponent);
  }

  public int getCategoryId() {
    return this.categoryId;
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
    builder.append("FruehwirthMultiParticle [linearState=")
        .append(this.linearState).append(", augResponseSample=")
        .append(this.augResponseSample).append(", predPriorObsMean=")
        .append(this.priorPredMean).append(", predPriorObsCov=")
        .append(this.priorPredCov).append(", categoryId=")
        .append(this.categoryId).append("]");
    return builder.toString();
  }
  
  
}
