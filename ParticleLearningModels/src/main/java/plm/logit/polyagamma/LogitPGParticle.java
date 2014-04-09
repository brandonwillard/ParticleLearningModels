package plm.logit.polyagamma;

import org.apache.commons.lang3.ObjectUtils;

import plm.logit.LogitParticle;

import com.statslibextensions.util.ComparableWeighted;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.ObjectUtil;
import gov.sandia.cognition.util.Weighted;

public class LogitPGParticle extends AbstractCloneableSerializable 
        implements ComparableWeighted, LogitParticle {

  protected LogitParticle previousParticle;
  protected KalmanFilter regressionFilter;
  protected MultivariateGaussian linearState;
  protected double priorPredMean;
  protected double priorPredCov;
  protected Vector augResponseSample = null;
  protected Vector betaSample = null;
  protected double logWeight;
  private double[] compLikelihoods;

  @Override
  public double getWeight() {
    return this.logWeight;
  }

  @Override
  public void setLogWeight(double logWeight) {
    this.logWeight = logWeight;
  }

  @Override
  public Vector getBetaSample() {
    return this.betaSample;
  }

  @Override
  public Vector getAugResponseSample() {
    return this.augResponseSample;
  }

  @Override
  public void setAugResponseSample(Vector augResponseSample) {
    this.augResponseSample = augResponseSample;
  }

  public LogitPGParticle(
      LogitPGParticle previousParticle, KalmanFilter linearComponent,
      MultivariateGaussian linearState) {
    this.previousParticle = previousParticle;
    this.regressionFilter = linearComponent;
    this.linearState = linearState;
  }

  @Override
  public LogitPGParticle clone() {
    LogitPGParticle clone = (LogitPGParticle) super.clone();
    clone.previousParticle = this.previousParticle;
    // when do we ever need a deep copy?  we don't alter
    // the components of a kalman filter in place...
    clone.regressionFilter = 
        new KalmanFilter(
            new LinearDynamicalSystem(
                this.regressionFilter.getModel().getA(),
                this.regressionFilter.getModel().getB(),
                this.regressionFilter.getModel().getC()), 
            this.regressionFilter.getModelCovariance(), 
            this.regressionFilter.getMeasurementCovariance());
    // same here
    clone.linearState = new MultivariateGaussian(
        this.linearState.getMean(), this.linearState.getCovariance());
    clone.augResponseSample = this.augResponseSample;
    clone.priorPredMean = this.priorPredMean;
    clone.priorPredCov = this.priorPredCov;
    clone.compLikelihoods = this.compLikelihoods;
    return clone;
  }

  @Override
  public MultivariateGaussian getLinearState() {
    return linearState;
  }

  @Override
  public void setLinearState(MultivariateGaussian linearState) {
    this.linearState = linearState;
  }

  @Override
  public LogitParticle getPreviousParticle() {
    return previousParticle;
  }

  @Override
  public void setPreviousParticle(LogitParticle previousParticle) {
    this.previousParticle = previousParticle;
  }

  @Override
  public KalmanFilter getRegressionFilter() {
    return regressionFilter;
  }

  public void setRegressionFilter(KalmanFilter linearComponent) {
    this.regressionFilter = linearComponent;
  }

  @Override
  public void setPriorPredMean(double predPriorObsMean) {
    this.priorPredMean = predPriorObsMean;
  }

  @Override
  public void setPriorPredCov(double predPriorObsCov) {
    this.priorPredCov = predPriorObsCov;
  }

  @Override
  public double getPriorPredMean() {
    return this.priorPredMean;
  }

  @Override
  public double getPriorPredCov() {
    return this.priorPredCov;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("LogitMixParticle [linearState=").append(this.linearState);
    builder
        .append("\t, predPriorObsMean=").append(this.priorPredMean)
        .append(", predPriorObsCov=").append(this.priorPredCov);
    if (this.betaSample != null)
        builder.append("\t, augResponseSample=").append(this.augResponseSample).append("\n");
    if (this.betaSample != null)
      builder.append("\t, betaSample=").append(this.betaSample).append("\n");
    builder.append("\t, weight=").append(this.logWeight);
    builder.append("]");
    return builder.toString();
  }

  @Override
  public void setBetaSample(Vector betaSample) {
    this.betaSample = betaSample;
  }

  @Override
  public int compareTo(Weighted o) {
    return Double.compare(this.getWeight(), o.getWeight());
  }
  
}
