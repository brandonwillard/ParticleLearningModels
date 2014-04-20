package plm.logit;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.CloneableSerializable;
import gov.sandia.cognition.util.Weighted;

public interface LogitParticle extends CloneableSerializable {


  public abstract void setLogWeight(double logWeight);

  public abstract Vector getBetaSample();

  public abstract Vector getAugResponseSample();

  public abstract void setAugResponseSample(Vector augResponseSample);

  public abstract MultivariateGaussian getLinearState();

  public abstract void setLinearState(MultivariateGaussian linearState);

  public abstract <T extends LogitParticle> T getPreviousParticle();

  public abstract void setPreviousParticle(LogitParticle previousParticle);

  public abstract KalmanFilter getRegressionFilter();

  public abstract void setRegressionFilter(KalmanFilter linearComponent);

  public abstract void setPriorPredMean(double predPriorObsMean);

  public abstract void setPriorPredCov(double predPriorObsCov);

  public abstract double getPriorPredMean();

  public abstract double getPriorPredCov();

  public abstract void setBetaSample(Vector betaSample);

}