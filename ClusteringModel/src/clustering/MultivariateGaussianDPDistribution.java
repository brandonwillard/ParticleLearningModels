package clustering;

import java.util.List;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.MultivariateMixtureDensityModel;
import gov.sandia.cognition.statistics.distribution.NormalInverseWishartDistribution;

/**
 * A multivariate Gaussian Dirichlet Process mixture distribution.
 * Essentially, a standard mixture with a concentration parameter.
 * @author bwillard
 *
 */
public class MultivariateGaussianDPDistribution extends
    MultivariateMixtureDensityModel<MultivariateGaussian> {
  
  private NormalInverseWishartDistribution centeringDistribution;
  private double dpAlpha;
  private Vector nCounts;
  private double index = 1d;
  
  /*
   * We hold onto the prior predictive log likelihoods for 
   * later sampling a component index (in the update step, after
   * resampling).
   */
  private double[] componentPriorPredLogLikelihoods = null;
  private double componentPriorPredTotalLogLikelihood = Double.NEGATIVE_INFINITY;

  public MultivariateGaussianDPDistribution(List<MultivariateGaussian> priorComponents,
    NormalInverseWishartDistribution normalInverseWishartDistribution, double dpAlphaPrior, Vector nCounts) {
    super(priorComponents, nCounts.scale(1d/nCounts.norm1()).toArray());
    this.centeringDistribution = normalInverseWishartDistribution;
    this.dpAlpha = dpAlphaPrior;
    this.nCounts = nCounts;
  }

  public NormalInverseWishartDistribution getCenteringDistribution() {
    return centeringDistribution;
  }

  public double[] getComponentPriorPredLogLikelihoods() {
    return componentPriorPredLogLikelihoods;
  }

  public void setCenteringDistribution(
    NormalInverseWishartDistribution centeringDistribution) {
    this.centeringDistribution = centeringDistribution;
  }

  public Vector getCounts() {
    return nCounts;
  }

  public void setnCounts(Vector nCounts) {
    this.priorWeights = nCounts.scale(1d/nCounts.norm1()).toArray();
    this.nCounts = nCounts;
  }

  public double getAlpha() {
    return dpAlpha;
  }

  public void setAlpha(double dpAlpha) {
    this.dpAlpha = dpAlpha;
  }

  public double getIndex() {
    return this.index;
  }
  
  public void setIndex(double index) {
    this.index = index;
  }

  public void setComponentPriorPredLogLikelihoods(
    double[] componentPriorPredLogLikelihoods, double totalLogLikelihood) {
    this.componentPriorPredLogLikelihoods = componentPriorPredLogLikelihoods;
    this.componentPriorPredTotalLogLikelihood  = totalLogLikelihood;
  }

  public double getComponentPriorPredTotalLogLikelihood() {
    return componentPriorPredTotalLogLikelihood;
  }

}