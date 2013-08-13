package clustering;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.MultivariateMixtureDensityModel;
import gov.sandia.cognition.statistics.distribution.NormalInverseWishartDistribution;

import java.util.List;

/**
 * A multivariate Gaussian Dirichlet Process mixture distribution. Important: the Gaussian
 * distributions have their covariances set to the sum of squares for their respective components.
 * 
 * @author bwillard
 * 
 */
public class MvGaussianDPDistribution extends MultivariateMixtureDensityModel<MultivariateGaussian> {

  private NormalInverseWishartDistribution centeringDistribution;
  private double dpAlpha;
  private Vector nCounts;
  private double index = 1d;

  /*
   * We hold onto the prior predictive log likelihoods for later sampling a component index (in the
   * update step, after resampling).
   */
  private double[] componentPriorPredLogLikelihoods = null;
  private double componentPriorPredTotalLogLikelihood = Double.NEGATIVE_INFINITY;

  public MvGaussianDPDistribution(List<MultivariateGaussian> priorComponents,
      NormalInverseWishartDistribution normalInverseWishartDistribution, double dpAlphaPrior,
      Vector nCounts) {
    super(priorComponents, nCounts.scale(1d / nCounts.norm1()).toArray());
    this.centeringDistribution = normalInverseWishartDistribution;
    this.dpAlpha = dpAlphaPrior;
    this.nCounts = nCounts;
  }

  public double getAlpha() {
    return dpAlpha;
  }

  public NormalInverseWishartDistribution getCenteringDistribution() {
    return centeringDistribution;
  }

  public double[] getComponentPriorPredLogLikelihoods() {
    return componentPriorPredLogLikelihoods;
  }

  public double getComponentPriorPredTotalLogLikelihood() {
    return componentPriorPredTotalLogLikelihood;
  }

  public Vector getCounts() {
    return nCounts;
  }

  public double getIndex() {
    return this.index;
  }

  public void setAlpha(double dpAlpha) {
    this.dpAlpha = dpAlpha;
  }

  public void setCenteringDistribution(NormalInverseWishartDistribution centeringDistribution) {
    this.centeringDistribution = centeringDistribution;
  }

  public void setComponentPriorPredLogLikelihoods(double[] componentPriorPredLogLikelihoods,
      double totalLogLikelihood) {
    this.componentPriorPredLogLikelihoods = componentPriorPredLogLikelihoods;
    this.componentPriorPredTotalLogLikelihood = totalLogLikelihood;
  }

  public void setIndex(double index) {
    this.index = index;
  }

  public void setnCounts(Vector nCounts) {
    this.priorWeights = nCounts.scale(1d / nCounts.norm1()).toArray();
    this.nCounts = nCounts;
  }

  @Override
  public String toString() {
    final StringBuilder retval = new StringBuilder(1000);
    retval.append("LinearMixtureModel has " + this.getDistributionCount() + " distributions:\n");
    int k = 0;
    for (final MultivariateGaussian distribution : this.getDistributions()) {
      retval.append(" " + k + ": Prior: " + this.getPriorWeights()[k] + ", Distribution:\nMean: "
          + distribution.getMean() + "\nCovariance:"
          + distribution.getCovariance().scale(1d / this.nCounts.getElement(k)) + "\n");
      k++;
    }
    return retval.toString();
  }

}
