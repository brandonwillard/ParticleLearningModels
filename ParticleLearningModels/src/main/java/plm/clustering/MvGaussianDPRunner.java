package plm.clustering;

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

import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;

/**
 * This class produces draws from a multivariate Gaussian mixture model and fits them with a
 * Dirichlet Process mixture of Gaussians using a Particle Learning filter. <br>
 * <br>
 * The model is from Example 2 of <a
 * href="http://faculty.chicagobooth.edu/nicholas.polson/research/papers/Bmix.pdf">
 * "Particle Learning for General Mixtures"</a>
 * 
 * @author bwillard
 * 
 */
public class MvGaussianDPRunner {

  public static void main(String[] args) {

    /*
     * Create a mixture distribution to fit.
     */
    final double[] trueComponentWeights = new double[] {0.1d, 0.5d, 0.1d, 0.2d, 0.1d};
    final List<MultivariateGaussian> trueComponentModels = Lists.newArrayList();
    trueComponentModels.add(new MultivariateGaussian(VectorFactory.getDenseDefault().copyArray(
        new double[] {0d, 0d}), MatrixFactory.getDenseDefault().copyArray(
        new double[][] { {100d, 0d}, {0d, 100d}})));
    trueComponentModels.add(new MultivariateGaussian(VectorFactory.getDenseDefault().copyArray(
        new double[] {100d, 1d}), MatrixFactory.getDenseDefault().copyArray(
        new double[][] { {100d, 0d}, {0d, 100d}})));
    trueComponentModels.add(new MultivariateGaussian(VectorFactory.getDenseDefault().copyArray(
        new double[] {10d, -10d}), MatrixFactory.getDenseDefault().copyArray(
        new double[][] { {100d, 0d}, {0d, 100d}})));
    trueComponentModels.add(new MultivariateGaussian(VectorFactory.getDenseDefault().copyArray(
        new double[] {1d, -200d}), MatrixFactory.getDenseDefault().copyArray(
        new double[][] { {10d, 0d}, {0d, 10d}})));
    trueComponentModels.add(new MultivariateGaussian(VectorFactory.getDenseDefault().copyArray(
        new double[] {50d, -50d}), MatrixFactory.getDenseDefault().copyArray(
        new double[][] { {20d, 0d}, {0d, 20d}})));

    final MultivariateMixtureDensityModel<MultivariateGaussian> trueMixture =
        new MultivariateMixtureDensityModel<MultivariateGaussian>(trueComponentModels,
            trueComponentWeights);

    final Random rng = new Random(829351983l);
    /*
     * Sample a lot of test data to fit against. TODO For a proper study, we would randomize subsets
     * of this data and fit against those.
     */
    final List<Vector> observations = trueMixture.sample(rng, 10000);

    /*
     * Instantiate PL filter by first providing prior parameters/distributions. We start by creating
     * a prior conjugate centering distribution (which is a Normal Inverse Wishart), then we provide
     * the Dirichlet Process prior parameters (group counts and concentration parameter).
     */
    final int centeringCovDof = 2 + 2;
    final Matrix centeringCovPriorMean =
        MatrixFactory.getDenseDefault().copyArray(new double[][] { {1000d, 0d}, {0d, 1000d}});
    final InverseWishartDistribution centeringCovariancePrior =
        new InverseWishartDistribution(centeringCovPriorMean.scale(centeringCovDof
            - centeringCovPriorMean.getNumColumns() - 1d), centeringCovDof);
    final MultivariateGaussian centeringMeanPrior =
        new MultivariateGaussian(VectorFactory.getDenseDefault().copyArray(new double[] {0d, 0d}),
            centeringCovariancePrior.getMean());
    final double centeringCovDivisor = 0.25d;
    final NormalInverseWishartDistribution centeringPrior =
        new NormalInverseWishartDistribution(centeringMeanPrior, centeringCovariancePrior,
            centeringCovDivisor);
    final double dpAlphaPrior = 2d;

    /*
     * Initialize the an empty mixture. The components will be created form the Dirichlet process
     * defined above.
     */
    final Vector nCountsPrior = VectorFactory.getDenseDefault().copyArray(new double[] {});
    final List<MultivariateGaussian> priorComponents = Lists.newArrayList();

    /*
     * Create and initialize the PL filter
     */
    final MvGaussianDPPLFilter plFilter =
        new MvGaussianDPPLFilter(priorComponents, centeringPrior, dpAlphaPrior, nCountsPrior, rng);
    plFilter.setNumParticles(500);

    final DataDistribution<MvGaussianDPDistribution> currentMixtureDistribution =
        plFilter.createInitialLearnedObject();
    for (final Vector observation : observations) {
      System.out.println("obs:" + observation);
      plFilter.update(currentMixtureDistribution, observation);

      /*
       * Compute some summary stats. TODO We need to compute something informative for this
       * situation.
       */
      final UnivariateGaussian.SufficientStatistic rmseSuffStat =
          new UnivariateGaussian.SufficientStatistic();
      final RingAccumulator<MutableDouble> countSummary = new RingAccumulator<>();

      for (final MvGaussianDPDistribution mixtureDist : currentMixtureDistribution.getDomain()) {
        rmseSuffStat.update(observation.minus(mixtureDist.getMean()).norm2());
        countSummary.accumulate(new MutableDouble(mixtureDist.getDistributionCount()));
      }
      System.out.println("posterior RMSE mean:" + rmseSuffStat.getMean());
      System.out.println("posterior component counts:" + countSummary.getMean());
    }

    System.out.println("finished simulation");

  }

}
