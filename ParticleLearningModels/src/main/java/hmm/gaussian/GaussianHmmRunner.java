package hmm.gaussian;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import hmm.HmmPlFilter;
import hmm.HmmResampleComparisonRunner;

import java.io.IOException;
import java.util.Random;

import com.google.common.collect.Lists;

public class GaussianHmmRunner extends HmmResampleComparisonRunner {

  public static void main(String[] args) throws IOException {

    final long seed = new Random().nextLong();
    final Random rng = new Random(seed);
    log.info("seed=" + seed);

    final int N = 100000;

    final UnivariateGaussian s1Likelihood = new UnivariateGaussian();
    final UnivariateGaussian s2Likelihood = s1Likelihood;

    final HiddenMarkovModel<Double> trueHmm1 =
        new HiddenMarkovModel<Double>(VectorFactory.getDefault()
            .copyArray(new double[] { 1d / 2d, 1d / 2d }),
            MatrixFactory.getDefault().copyArray(
                new double[][] { { 1d / 2d, 1d / 2d },
                    { 1d / 2d, 1d / 2d } }), Lists.newArrayList(
                s1Likelihood, s2Likelihood));
    final HmmPlFilter<Double> wfFilter =
        new GaussianHmmPlFilter(trueHmm1, rng, false);
    final HmmPlFilter<Double> rsFilter =
        new GaussianHmmPlFilter(trueHmm1, rng, true);
    waterFillResampleComparison(wfFilter, rsFilter, trueHmm1, N, 60,
        10, "hmm-nig-wf-rs-10000-class-errors-m1.csv", rng);

    final HiddenMarkovModel<Double> trueHmm2 =
        new HiddenMarkovModel<Double>(VectorFactory.getDefault()
            .copyArray(new double[] { 1d / 3d, 2d / 3d }),
            MatrixFactory.getDefault().copyArray(
                new double[][] { { 9d / 10d, 1d / 10d },
                    { 1d / 10d, 9d / 10d } }), Lists.newArrayList(
                s1Likelihood, s2Likelihood));
    final HmmPlFilter<Double> wfFilter2 =
        new GaussianHmmPlFilter(trueHmm2, rng, false);
    final HmmPlFilter<Double> rsFilter2 =
        new GaussianHmmPlFilter(trueHmm2, rng, true);
    waterFillResampleComparison(wfFilter2, rsFilter2, trueHmm2, N,
        60, 10, "hmm-nig-wf-rs-10000-class-errors-m2.csv", rng);

  }
}
