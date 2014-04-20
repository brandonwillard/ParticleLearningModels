package plm.logit.fruehwirth;

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

public class LogitMixParticle extends AbstractCloneableSerializable 
        implements ComparableWeighted, LogitParticle {

  protected UnivariateGaussian EVcomponent;
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

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#getBetaSample()
   */
  @Override
  public Vector getBetaSample() {
    return this.betaSample;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#getAugResponseSample()
   */
  @Override
  public Vector getAugResponseSample() {
    return this.augResponseSample;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#setAugResponseSample(gov.sandia.cognition.math.matrix.Vector)
   */
  @Override
  public void setAugResponseSample(Vector augResponseSample) {
    this.augResponseSample = augResponseSample;
  }

  public LogitMixParticle(
      LogitParticle previousParticle, KalmanFilter linearComponent,
      MultivariateGaussian linearState, UnivariateGaussian EVcomponent) {
    this.previousParticle = previousParticle;
    this.regressionFilter = linearComponent;
    this.linearState = linearState;
    this.EVcomponent = EVcomponent;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#clone()
   */
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

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#getLinearState()
   */
  @Override
  public MultivariateGaussian getLinearState() {
    return linearState;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#setLinearState(gov.sandia.cognition.statistics.distribution.MultivariateGaussian)
   */
  @Override
  public void setLinearState(MultivariateGaussian linearState) {
    this.linearState = linearState;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#getPreviousParticle()
   */
  @Override
  public LogitParticle getPreviousParticle() {
    return previousParticle;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#setPreviousParticle(plm.logit.fruehwirth.LogitParticle)
   */
  @Override
  public void setPreviousParticle(LogitParticle previousParticle) {
    this.previousParticle = previousParticle;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#getRegressionFilter()
   */
  @Override
  public KalmanFilter getRegressionFilter() {
    return regressionFilter;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#setRegressionFilter(gov.sandia.cognition.statistics.bayesian.KalmanFilter)
   */
  @Override
  public void setRegressionFilter(KalmanFilter linearComponent) {
    this.regressionFilter = linearComponent;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#setPriorPredMean(double)
   */
  @Override
  public void setPriorPredMean(double predPriorObsMean) {
    this.priorPredMean = predPriorObsMean;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#setPriorPredCov(double)
   */
  @Override
  public void setPriorPredCov(double predPriorObsCov) {
    this.priorPredCov = predPriorObsCov;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#getPriorPredMean()
   */
  @Override
  public double getPriorPredMean() {
    return this.priorPredMean;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#getPriorPredCov()
   */
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

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#setBetaSample(gov.sandia.cognition.math.matrix.Vector)
   */
  @Override
  public void setBetaSample(Vector betaSample) {
    this.betaSample = betaSample;
  }

  /* (non-Javadoc)
   * @see plm.logit.fruehwirth.LogitParticle#compareTo(gov.sandia.cognition.util.Weighted)
   */
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
