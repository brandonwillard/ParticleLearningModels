package plm.hmm.categorical;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;

import java.io.IOException;
import java.util.Random;

import plm.hmm.HmmPlFilter;
import plm.hmm.HmmResampleComparisonRunner;
import plm.hmm.HmmTransitionState;
import plm.hmm.StandardHMM;

import com.google.common.collect.Lists;

public class CategoricalHmmRunner extends HmmResampleComparisonRunner {

  public static void main(String[] args) throws IOException {

    final long seed = new Random().nextLong();
    final Random rng = new Random(seed);
    log.info("seed=" + seed);

    final int N = 10;

    DefaultDataDistribution<Integer> s1Likelihood = new DefaultDataDistribution<Integer>();
    s1Likelihood.increment(0, 1d/2d);
    s1Likelihood.increment(1, 1d/2d);
    DefaultDataDistribution<Integer> s2Likelihood = s1Likelihood;

    final StandardHMM<Integer> trueHmm1 =
        StandardHMM.create(
        new HiddenMarkovModel<Integer>(VectorFactory.getDefault()
            .copyArray(new double[] { 1d / 2d, 1d / 2d }),
            MatrixFactory.getDefault().copyArray(
                new double[][] { { 1d / 2d, 1d / 2d },
                    { 1d / 2d, 1d / 2d } }), Lists.newArrayList(
                s1Likelihood, s2Likelihood)));

    final HmmPlFilter<StandardHMM<Integer>, HmmTransitionState<Integer, StandardHMM<Integer>>, Integer> wfFilter =
        new CategoricalHmmPlFilter(trueHmm1, rng, false);
    final HmmPlFilter<StandardHMM<Integer>, HmmTransitionState<Integer, StandardHMM<Integer>>, Integer> rsFilter =
        new CategoricalHmmPlFilter(trueHmm1, rng, true);
    waterFillResampleComparison(wfFilter, rsFilter, trueHmm1, N, 60,
        10, "hmm-cat-wf-rs-10000-class-errors-m1.csv", rng);

    final StandardHMM<Integer> trueHmm2 =
        StandardHMM.create(
        new HiddenMarkovModel<Integer>(VectorFactory.getDefault()
            .copyArray(new double[] { 1d / 3d, 2d / 3d }),
            MatrixFactory.getDefault().copyArray(
                new double[][] { { 9d / 10d, 1d / 10d },
                    { 1d / 10d, 9d / 10d } }), Lists.newArrayList(
                s1Likelihood, s2Likelihood)));
    final HmmPlFilter<StandardHMM<Integer>, HmmTransitionState<Integer, StandardHMM<Integer>>, Integer> wfFilter2 =
        new CategoricalHmmPlFilter(trueHmm2, rng, false);
    final HmmPlFilter<StandardHMM<Integer>, HmmTransitionState<Integer, StandardHMM<Integer>>, Integer> rsFilter2 =
        new CategoricalHmmPlFilter(trueHmm2, rng, true);
    waterFillResampleComparison(wfFilter2, rsFilter2, trueHmm2, N,
        60, 10, "hmm-cat-wf-rs-10000-class-errors-m2.csv", rng);
  }
    
}
