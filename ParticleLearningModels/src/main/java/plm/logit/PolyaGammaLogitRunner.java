package plm.logit;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;
import com.statslibextensions.statistics.distribution.ScaledInverseGammaCovDistribution;
import com.statslibextensions.util.ObservedValue;

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
public class PolyaGammaLogitRunner {

  public static void main(String[] args) {


    final double trueGlobalMean = 30d;
    final Vector trueBetas =
        VectorFactory.getDenseDefault().copyArray(new double[] {0d});

    final Random rng = new Random(829351983l);
    /*
     * Sample test data from a binomial with log odds defined using the above parameters. Also, we
     * produce predictor values by generating the randomly.
     */
//    final MultivariateGaussian dataGeneratingDist =
//        new MultivariateGaussian(VectorFactory.getDenseDefault().copyArray(
//            new double[] {100d, -50d, 34d, 0d, 1e-3d}), MatrixFactory.getDenseDefault()
//            .createDiagonal(
//                VectorFactory.getDenseDefault().copyArray(new double[] {100d, 10d, 500d, 30d, 1d})));
    final List<ObservedValue> observations = Lists.newArrayList();
    for (int i = 0; i < 10000; i++) {
//      final Vector dataSample = dataGeneratingDist.sample(rng);
      final Vector dataSample = VectorFactory.getDenseDefault().copyArray(new 
        double[] {1d});
      final double phi = Math.exp(-trueGlobalMean - dataSample.dotProduct(trueBetas));
      final double pi = 1d / (1d + phi);
      final Vector y = VectorFactory.getDenseDefault().createVector1D(rng.nextDouble() <= pi ? 1d : 0d);
      final Matrix dataDesign = MatrixFactory.getDenseDefault().copyRowVectors(dataSample);
      observations.add(new ObservedValue(i, y, dataDesign));
    }

    /*
     * Instantiate PL filter by first providing prior parameters/distributions.
     */

    final Vector betaCovPriorMean = VectorFactory.getDefault().copyArray(
      new double[] {200d});
    final double betaPriorCovDof = 2 + betaCovPriorMean.getDimensionality();
    final ScaledInverseGammaCovDistribution priorBetaCov =
        new ScaledInverseGammaCovDistribution(betaCovPriorMean.getDimensionality(),  
          betaCovPriorMean.scale(betaPriorCovDof - 1d).getElement(0), 
          betaPriorCovDof);
    final MultivariateGaussian priorBeta =
        new MultivariateGaussian(VectorFactory.getDenseDefault().copyArray(
            new double[] {0d}), priorBetaCov.getMean());

    /*
     * Create and initialize the PL filter
     */
    final PolyaGammaLogitPLFilter plFilter =
        new PolyaGammaLogitPLFilter(rng, priorBeta, priorBetaCov);
    plFilter.setNumParticles(500);

    final DataDistribution<PolyaGammaLogitDistribution> currentMixtureDistribution =
        plFilter.createInitialLearnedObject();
    for (final ObservedValue observation : observations) {
      System.out.println("obs:" + observation);
      plFilter.update(currentMixtureDistribution, observation);

      /*
       * Compute some summary stats. TODO We need to compute something informative for this
       * situation.
       */
      // UnivariateGaussian.SufficientStatistic rmseSuffStat = new
      // UnivariateGaussian.SufficientStatistic();
      // RingAccumulator<MutableDouble> countSummary = new RingAccumulator<>();
      //
      // for (MvGaussianDPDistribution mixtureDist : currentMixtureDistribution.getDomain()) {
      // rmseSuffStat.update(observation.minus(mixtureDist.getMean()).norm2());
      // countSummary.accumulate(new MutableDouble(mixtureDist.getDistributionCount()));
      // }
      // System.out.println("posterior RMSE mean:" + rmseSuffStat.getMean());
      // System.out.println("posterior component counts:" + countSummary.getMean());
    }

    System.out.println("finished simulation");

  }

}
