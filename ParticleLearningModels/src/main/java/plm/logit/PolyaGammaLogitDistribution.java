package plm.logit;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.Distribution;
import gov.sandia.cognition.statistics.distribution.ExponentialDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;
import com.statslibextensions.statistics.distribution.PolyaGammaDistribution;
import com.statslibextensions.statistics.distribution.ScaledInverseGammaCovDistribution;
import com.statslibextensions.util.ExtStatisticsUtils;

/**
 * 
 * A binary logit model with linear gaussian regression log-odds and Lasso shrinkage.
 * 
 * 
 * @author bwillard
 * 
 */
public class PolyaGammaLogitDistribution extends PolyaGammaDistribution {


  /*
   * The joint distribution of augmented response (based on indep. PG(1, 0))
   */
  protected MultivariateGaussian augmentedResponseDistribution;

  /*
   * Regression coefficients
   */
  protected MultivariateGaussian priorBeta;

  protected ScaledInverseGammaCovDistribution priorBetaCov;

  /*
   * Keep this value around for debugging.
   */
  protected Vector priorPredictiveMean;

  public PolyaGammaLogitDistribution(MultivariateGaussian priorBeta,
      ScaledInverseGammaCovDistribution scaledInverseGammaCovDistribution) {
    super(Double.NaN);
    this.priorBeta = priorBeta;
    this.priorBetaCov = scaledInverseGammaCovDistribution;
  }

  public MultivariateGaussian getAugmentedResponseDistribution() {
    return augmentedResponseDistribution;
  }

  public MultivariateGaussian getPriorBeta() {
    return priorBeta;
  }

  public ScaledInverseGammaCovDistribution getPriorBetaCov() {
    return priorBetaCov;
  }

  public Vector getPriorPredictiveMean() {
    return this.priorPredictiveMean;
  }

  public void setAugmentedResponseDistribution(MultivariateGaussian augmentedResponseDistribution) {
    this.augmentedResponseDistribution = augmentedResponseDistribution;
  }

  public void setPriorBeta(MultivariateGaussian priorBeta) {
    this.priorBeta = priorBeta;
  }

  public void setPriorBetaCov(ScaledInverseGammaCovDistribution priorBetaCov) {
    this.priorBetaCov = priorBetaCov;
  }

  public void setPriorPredictiveMean(Vector phi) {
    this.priorPredictiveMean = phi;
  }

  @Override
  public ArrayList<? extends Double> sample(Random random, int numSamples) {
    List<Double> samples = Lists.newArrayList();
    for (int i = 0; i < numSamples; i++) {
      samples.add(this.sample(random));
    }
    return (ArrayList<? extends Double>) samples;
  }

}
