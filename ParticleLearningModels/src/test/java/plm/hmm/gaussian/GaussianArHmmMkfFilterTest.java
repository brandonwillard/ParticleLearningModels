package plm.hmm.gaussian;

import static org.junit.Assert.*;
import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;

import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;
import org.junit.Test;

import plm.hmm.DlmHiddenMarkovModel;
import plm.hmm.GenericHMM.SimHmmObservedValue;
import plm.hmm.HmmResampleComparisonRunner;
import plm.hmm.StandardHMM;

import com.google.common.collect.Lists;
import com.statslibextensions.statistics.CountedDataDistribution;
import com.statslibextensions.util.ObservedValue;

public class GaussianArHmmMkfFilterTest {

  protected static final Logger log = Logger
        .getLogger(HmmResampleComparisonRunner.class);

  @Test
  public void test() {
 
    final long seed = new Random().nextLong();
    final Random random = new Random(seed);
    log.info("seed=" + seed);

    final int N = 10;

    final double[] a = {0.9d, 0.9d};
    final double[] sigma2 = {Math.pow(0.2d, 2), Math.pow(1.2d, 2)};
    final double sigma_y2 = Math.pow(0.3d, 2);
    Matrix modelCovariance1 = MatrixFactory.getDefault().copyArray(
        new double[][] {{Math.pow(0.2d, 2)}});
    Matrix modelCovariance2 = MatrixFactory.getDefault().copyArray(
        new double[][] {{Math.pow(1.2d, 2)}});
    Matrix measurementCovariance = MatrixFactory.getDefault().copyArray(
        new double[][] {{Math.pow(0.3d, 2)}});
    LinearDynamicalSystem model1 = new LinearDynamicalSystem(
        MatrixFactory.getDefault().copyArray(new double[][] {{0.9d}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1}})
      );
    LinearDynamicalSystem model2 = new LinearDynamicalSystem(
        MatrixFactory.getDefault().copyArray(new double[][] {{0.9d}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1}})
      );
    KalmanFilter trueKf1 = new KalmanFilter(model1, modelCovariance1, measurementCovariance);
    KalmanFilter trueKf2 = new KalmanFilter(model2, modelCovariance2, measurementCovariance);

    final UnivariateGaussian prior = new UnivariateGaussian(0d, sigma_y2);
    final UnivariateGaussian s1Likelihood = prior;
    final UnivariateGaussian s2Likelihood = s1Likelihood;
    
    Vector initialClassProbs = VectorFactory.getDefault()
            .copyArray(new double[] { 0.7d, 0.3d });
    Matrix classTransProbs = MatrixFactory.getDefault().copyArray(
                new double[][] { { 0.7d, 0.7d },
                    { 0.3d, 0.3d } });
    
    DlmHiddenMarkovModel dlmHmm = new DlmHiddenMarkovModel(
        Lists.newArrayList(trueKf1, trueKf2), 
        initialClassProbs, classTransProbs);

    final StandardHMM<Double> trueHmm1 =
        StandardHMM.create(
        new HiddenMarkovModel<Double>(initialClassProbs,
            classTransProbs, Lists.newArrayList(
                s1Likelihood, s2Likelihood)));

    final GaussianArHmmMkfFilter rsFilter =
        new GaussianArHmmMkfFilter(trueHmm1, prior, a, sigma2, sigma_y2, random);

    final int K = 3;
    final int T = 100;

    List<SimHmmObservedValue<Vector, Vector>> simulation = dlmHmm.sample(random, T);

    rsFilter.setNumParticles(N);

    GaussianArHmmRmseEvaluator mkfRmseEvaluator = new GaussianArHmmRmseEvaluator("mkf", 
        null);

    for (int k = 0; k < K; k++) {
      log.info("Processing replication " + k);

      DataDistribution<GaussianArTransitionState> rsDistribution =
          rsFilter.getUpdater().createInitialParticles(N);

      /*
       * Recurse through the particle filter
       */
      for (int i = 0; i < T; i++) {
  
        final double x = simulation.get(i).getClassId();
        final Double y = simulation.get(i).getObservedValue().getElement(0);
        // lame hack need until I refactor to use DlmHiddenMarkovModel in the filters
        final ObservedValue<Double> obsState = new SimHmmObservedValue<Double, Double>(i, 
           (int)x , simulation.get(i).getState().getElement(0), y);

        if (i > 0) {
          rsFilter.update(rsDistribution, obsState);
          mkfRmseEvaluator.evaluate(k, simulation.get(i), rsDistribution);
        }

        if ((i+1) % (T/4d) < 1) {
          log.info("avg. mkfRmse=" + mkfRmseEvaluator.getTotalRate().getMean().value);
        }
      }
      assertEquals(0d, mkfRmseEvaluator.getTotalRate().getMean().value, 9e-1);
    }
  }

}
