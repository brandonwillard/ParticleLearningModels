package plm.regularization.lasso;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.Distribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;

import java.util.ArrayList;
import java.util.Random;

import com.statslibextensions.statistics.distribution.ScaledInverseGammaCovDistribution;

/**
 * 
 * Linear regression with and bayesian lasso shrinkage (via double exponential marginal priors).
 * 
 * @author bwillard
 * 
 */
public class LassoRegressionDistribution extends AbstractCloneableSerializable implements Distribution<Vector> {


  /*
   * Regression coefficients
   */
  protected MultivariateGaussian priorBeta;

  protected ScaledInverseGammaCovDistribution priorObsCov;

  protected Matrix augLassoSample;

  protected Matrix priorObsCovSample;

  public LassoRegressionDistribution(MultivariateGaussian priorBeta,
      ScaledInverseGammaCovDistribution priorObsCov, Matrix augLassoSample,
      Matrix priorObsCovSample) {
    this.priorBeta = priorBeta;
    this.priorObsCov = priorObsCov;
    this.augLassoSample = augLassoSample;
    this.priorObsCovSample = priorObsCovSample;
  }

  public MultivariateGaussian getPriorBeta() {
    return priorBeta;
  }

  public ScaledInverseGammaCovDistribution getPriorObsCov() {
    return priorObsCov;
  }

  public Matrix getPriorObsCovSample() {
    return this.priorObsCovSample;
  }

  public void setPriorBeta(MultivariateGaussian priorBeta) {
    this.priorBeta = priorBeta;
  }

  public void setPriorObsCov(ScaledInverseGammaCovDistribution priorObsCov) {
    this.priorObsCov = priorObsCov;
  }

  /**
   * @see LassoRegressionDistribution#sample(Random)
   */
  @Override
  public ArrayList<Vector> sample(Random random, int numSamples) {
    return this.priorBeta.sample(random, numSamples);
  }

  /**
   * TODO FIXME: do we sample from both priors?  currently only beta.
   */
  @Override
  public Vector sample(Random random) {
    return this.priorBeta.sample(random);
  }

  public Matrix getAugLassoSample() {
    return augLassoSample;
  }

  public void setAugLassoSample(Matrix augLassoSample) {
    this.augLassoSample = augLassoSample;
  }

  public void setPriorObsCovSample(Matrix priorObsCovSample) {
    this.priorObsCovSample = priorObsCovSample;
  }

}
