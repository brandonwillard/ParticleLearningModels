package plm.ensemble;

import com.statslibextensions.util.ObservedValue;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.CloneableSerializable;

public class GaussianHpEnsParticle extends GaussianEnsParticle {

  InverseGammaDistribution sigma2SS;
  MultivariateGaussian psiSS;
  double sigma2Sample;
  private Vector psiSample;

  public GaussianHpEnsParticle(KalmanFilter thisKf,
      ObservedValue<Vector, ?> create, MultivariateGaussian priorState,
      Vector priorStateSample, InverseGammaDistribution priorSigma2,
      MultivariateGaussian priorPsi, double sigma2Sample, Vector psiPriorSmpl) {
    this(null, thisKf, create, 
        priorState, priorStateSample, 
        priorSigma2, priorPsi, 
        sigma2Sample, psiPriorSmpl);
  }

  public GaussianHpEnsParticle(GaussianHpEnsParticle prevParticle,
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

  public Vector getPsiSample() {
    return psiSample;
  }

  public InverseGammaDistribution getSigma2SS() {
    return this.sigma2SS;
  }

  public MultivariateGaussian getPsiSS() {
    return this.psiSS;
  }

  public double getSigma2Sample() {
    return this.sigma2Sample;
  }

  @Override
  public GaussianHpEnsParticle clone() {
    GaussianHpEnsParticle clone = (GaussianHpEnsParticle) super.clone();
    clone.sigma2Sample = this.sigma2Sample;
    clone.sigma2SS = this.sigma2SS.clone();
    clone.psiSS = this.psiSS.clone();
    return clone;
  }
}
