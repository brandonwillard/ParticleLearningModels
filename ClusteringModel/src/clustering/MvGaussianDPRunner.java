package clustering;

import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;

import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.MultivariateMixtureDensityModel;
import gov.sandia.cognition.statistics.distribution.NormalInverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;

public class MvGaussianDPRunner {

	public static void main(String[] args) {
	  
	  /*
	   * Create a mixture distribution to fit.
	   */
	  double[] trueComponentWeights = new double[] {0.1d, 0.5, 0.4};
	  List<MultivariateGaussian> trueComponentModels = Lists.newArrayList();
	  trueComponentModels.add(new MultivariateGaussian(
	      VectorFactory.getDenseDefault().copyArray(new double[] {0d, 0d}),
	      MatrixFactory.getDenseDefault().copyArray(new double[][] {
	          {100d, 0d},
	          {0d, 100d}
	      }))); 
	  trueComponentModels.add(new MultivariateGaussian(
	      VectorFactory.getDenseDefault().copyArray(new double[] {100d, 1d}),
	      MatrixFactory.getDenseDefault().copyArray(new double[][] {
	          {100d, 0d},
	          {0d, 100d}
	      }))); 
	  trueComponentModels.add(new MultivariateGaussian(
	      VectorFactory.getDenseDefault().copyArray(new double[] {10d, -10d}),
	      MatrixFactory.getDenseDefault().copyArray(new double[][] {
	          {100d, 0d},
	          {0d, 100d}
	      }))); 
	  MultivariateMixtureDensityModel<MultivariateGaussian> trueMixture = new MultivariateMixtureDensityModel<MultivariateGaussian>(
	      trueComponentModels, trueComponentWeights);
	  
	  Random rng = new Random(829351983l);
	  List<Vector> observations = trueMixture.sample(rng, 100);
	  
	  /*
	   * Instantiate PL filter by first providing prior parameters/distributions. 
	   * We start by creating a prior conjugate centering distribution, then 
	   * we provide the Dirichlet Process prior parameters (group counts and
	   * concentration parameter).
	   */
	  final int centeringCovDof  = 2 + 2;
	  Matrix centeringCovPriorMean = MatrixFactory.getDenseDefault().copyArray(new double[][] {
	          {1d, 0d},
	          {0d, 1d}
	      }).scale(1d/(centeringCovDof - 0.5d * (2d + 1d)));
	  InverseWishartDistribution centeringCovariancePrior = new InverseWishartDistribution(
	      centeringCovPriorMean.scale(centeringCovDof - centeringCovPriorMean.getNumColumns() - 1d), 
	      centeringCovDof);
	  MultivariateGaussian centeringMeanPrior = new MultivariateGaussian(
	      VectorFactory.getDenseDefault().copyArray(new double[] {0d, 0d}),
	      centeringCovariancePrior.getMean());
	  final double centeringCovDivisor = 0.25d;
	  NormalInverseWishartDistribution centeringPrior = new NormalInverseWishartDistribution(
	      centeringMeanPrior, centeringCovariancePrior, centeringCovDivisor);
	  final double dpAlphaPrior = 2d;
	  Vector nCountsPrior = VectorFactory.getDenseDefault().copyArray(new double[] {
	      1, 1
	  });
	  List<MultivariateGaussian> priorComponents = Lists.newArrayList(); 
	  priorComponents.add(new MultivariateGaussian(
	      VectorFactory.getDenseDefault().copyArray(new double[] {70d, 10d}),
	      MatrixFactory.getDenseDefault().copyArray(new double[][] {
	          {100d, 0d},
	          {0d, 100d}
	      }))); 
	  priorComponents.add(new MultivariateGaussian(
	      VectorFactory.getDenseDefault().copyArray(new double[] {10d, -10d}),
	      MatrixFactory.getDenseDefault().copyArray(new double[][] {
	          {100d, 0d},
	          {0d, 100d}
	      }))); 
	  
	  /*
	   * Create and initialize the PL filter
	   */
	  MvGaussianDPPLFilter plFilter = new MvGaussianDPPLFilter(
	      priorComponents, centeringPrior, dpAlphaPrior, nCountsPrior, rng);
	  plFilter.setNumParticles(100);
	  
	  DataDistribution<MvGaussianDPDistribution> currentMixtureDistribution =
	      plFilter.createInitialLearnedObject();
	  for (Vector observation : observations) {
	    System.out.println("obs:" + observation);
	    plFilter.update(currentMixtureDistribution, observation);
	    
	    /*
	     * Compute RMSE summary stats
	     */
	    UnivariateGaussian.SufficientStatistic rmseSuffStat = new UnivariateGaussian.SufficientStatistic();
	    RingAccumulator<MutableDouble> countSummary = new RingAccumulator<>();
	    
  	  for (MvGaussianDPDistribution mixtureDist : currentMixtureDistribution.getDomain()) {
  	    rmseSuffStat.update(observation.minus(mixtureDist.getMean()).norm2());
  	    countSummary.accumulate(new MutableDouble(mixtureDist.getDistributionCount()));
  	  }
//  	  BayesianCredibleInterval ci = BayesianCredibleInterval.compute(rmseSuffStat.create(), 0.97d);
//	    System.out.println("posterior RMSE 97% credible interval:" + ci);
	    System.out.println("posterior RMSE mean:" + rmseSuffStat.getMean());
	    System.out.println("posterior component counts:" + countSummary.getMean());
	  }

	}

}
