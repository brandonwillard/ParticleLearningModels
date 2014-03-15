package plm.logit.fruehwirth;

import org.apache.commons.lang3.ObjectUtils;

import com.statslibextensions.util.ComparableWeighted;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.ObjectUtil;
import gov.sandia.cognition.util.Weighted;

public class LogitMixParticle extends AbstractCloneableSerializable 
        implements ComparableWeighted {

  protected UnivariateGaussian EVcomponent;
  protected LogitMixParticle previousParticle;
  protected KalmanFilter regressionFilter;
  protected MultivariateGaussian linearState;
  protected double priorPredMean;
  protected double priorPredCov;
  protected Vector augResponseSample = null;
  protected Vector betaSample = null;
  protected double logWeight;
  private double[] compLikelihoods;

  /**
   * Get log weight
   * @return
   */
  public double getWeight() {
    return this.logWeight;
  }

  /**
   * Set log weight
   * @param logWeight
   */
  public void setWeight(double logWeight) {
    this.logWeight = logWeight;
  }

  public Vector getBetaSample() {
    return this.betaSample;
  }

  public Vector getAugResponseSample() {
    return this.augResponseSample;
  }

  public void setAugResponseSample(Vector augResponseSample) {
    this.augResponseSample = augResponseSample;
  }

  public LogitMixParticle(
      LogitMixParticle previousParticle, KalmanFilter linearComponent,
      MultivariateGaussian linearState, UnivariateGaussian EVcomponent) {
    this.previousParticle = previousParticle;
    this.regressionFilter = linearComponent;
    this.linearState = linearState;
    this.EVcomponent = EVcomponent;
  }

  @Override
  public LogitMixParticle clone() {
    LogitMixParticle clone = (LogitMixParticle) super.clone();
    clone.EVcomponent = this.EVcomponent;
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

  public LogitMixParticle getPreviousParticle() {
    return previousParticle;
  }

  public void setPreviousParticle(LogitMixParticle previousParticle) {
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
    builder.append("LogitMixParticle [linearState=").append(this.linearState);
    builder
        .append("\t, predPriorObsMean=").append(this.priorPredMean)
        .append(", predPriorObsCov=").append(this.priorPredCov).append("\n")
        .append("\t, evComponent=").append(this.EVcomponent).append("\n");
    if (this.betaSample != null)
        builder.append("\t, augResponseSample=").append(this.augResponseSample).append("\n");
    if (this.betaSample != null)
      builder.append("\t, betaSample=").append(this.betaSample).append("\n");
    builder.append("\t, weight=").append(this.logWeight);
    builder.append("]");
    return builder.toString();
  }

  public void setBetaSample(Vector betaSample) {
    this.betaSample = betaSample;
  }

  @Override
  public int compareTo(Weighted o) {
    return Double.compare(this.getWeight(), o.getWeight());
  }

  public void setComponentLikelihoods(double[] componentLikelihoods) {
    this.compLikelihoods = componentLikelihoods;
  }
  
  public double[] getComponentLikelihoods() {
    return this.compLikelihoods;
  }

  
}
