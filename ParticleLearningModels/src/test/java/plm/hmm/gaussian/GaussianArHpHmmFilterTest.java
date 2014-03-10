package plm.hmm.gaussian;

import static org.junit.Assert.*;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.io.FileWriter;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;
import org.junit.Test;

import plm.hmm.DlmHiddenMarkovModel;
import plm.hmm.GenericHMM.SimHmmObservedValue;
import plm.hmm.HmmPlFilter;
import plm.util.hmm.HmmResampleComparisonRunner;
import plm.util.hmm.gaussian.GaussianArHmmClassEvaluator;
import plm.util.hmm.gaussian.GaussianArHmmPsiLearningEvaluator;
import plm.util.hmm.gaussian.GaussianArHmmRmseEvaluator;
import au.com.bytecode.opencsv.CSVWriter;

import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;

public class GaussianArHpHmmFilterTest {

  protected static final Logger log = Logger
        .getLogger(HmmResampleComparisonRunner.class);

  @Test
  public void test() {

    final long seed = new Random().nextLong();
    final Random random = new Random(seed);
    log.info("seed=" + seed);

    final double trueSigma = Math.pow(0.2d, 2);
    Matrix modelCovariance1 = MatrixFactory.getDefault().copyArray(
        new double[][] {{trueSigma}});
    Matrix modelCovariance2 = MatrixFactory.getDefault().copyArray(
        new double[][] {{trueSigma}});
    Matrix measurementCovariance = MatrixFactory.getDefault().copyArray(
        new double[][] {{trueSigma}});

    List<Vector> truePsis = Lists.newArrayList(
        VectorFactory.getDefault().copyValues(3d, 0.2d),
        VectorFactory.getDefault().copyValues(1d, 0.9d));

    LinearDynamicalSystem model1 = new LinearDynamicalSystem(
        MatrixFactory.getDefault().copyArray(new double[][] {{truePsis.get(0).getElement(1)}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}})
      );
    LinearDynamicalSystem model2 = new LinearDynamicalSystem(
        MatrixFactory.getDefault().copyArray(new double[][] {{truePsis.get(1).getElement(1)}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}})
      );
    KalmanFilter trueKf1 = new KalmanFilter(model1, modelCovariance1, measurementCovariance);
    trueKf1.setCurrentInput(VectorFactory.getDefault().copyValues(truePsis.get(0).getElement(0)));
    KalmanFilter trueKf2 = new KalmanFilter(model2, modelCovariance2, measurementCovariance);
    trueKf2.setCurrentInput(VectorFactory.getDefault().copyValues(truePsis.get(1).getElement(0)));
    
    Vector initialClassProbs = VectorFactory.getDefault()
            .copyArray(new double[] { 0.7d, 0.3d });
    Matrix classTransProbs = MatrixFactory.getDefault().copyArray(
                new double[][] { { 0.7d, 0.7d },
                    { 0.3d, 0.3d } });
    
    DlmHiddenMarkovModel trueHmm1 = new DlmHiddenMarkovModel(
        Lists.newArrayList(trueKf1, trueKf2), 
        initialClassProbs, classTransProbs);

    final double sigmaPriorMean = Math.pow(0.4, 2);
    final double sigmaPriorShape = 2d;
    final double sigmaPriorScale = sigmaPriorMean*(sigmaPriorShape + 1d);
    final InverseGammaDistribution sigmaPrior = new InverseGammaDistribution(sigmaPriorShape,
        sigmaPriorScale);
    
    final Vector phiMean1 = VectorFactory.getDefault().copyArray(new double[] {
        0d, 0.8d
    });
    final Matrix phiCov1 = MatrixFactory.getDefault().copyArray(new double[][] {
        {2d + 4d * sigmaPriorMean, 0d},
        { 0d, 4d * sigmaPriorMean}
    });
    final MultivariateGaussian phiPrior1 = new MultivariateGaussian(phiMean1, phiCov1);

    final Vector phiMean2 = VectorFactory.getDefault().copyArray(new double[] {
        0d, 0.1d
    });
    final Matrix phiCov2 = MatrixFactory.getDefault().copyArray(new double[][] {
        { 1d + 4d * sigmaPriorMean, 0d},
        { 0d, 4d * sigmaPriorMean}
    });
    final MultivariateGaussian phiPrior2 = new MultivariateGaussian(phiMean2, phiCov2);
    
    List<MultivariateGaussian> priorPhis = Lists.newArrayList(phiPrior1, phiPrior2);

    final HmmPlFilter<DlmHiddenMarkovModel, GaussianArHpTransitionState, Vector> wfFilter =
        new GaussianArHpHmmPlFilter(trueHmm1, sigmaPrior, priorPhis, random, true);

    final int K = 3;
    final int T = 200;
    final int N = 1000;

    /*
     * Note: replications are over the same set of simulated observations.
     */
    List<SimHmmObservedValue<Vector, Vector>> simulation = trueHmm1.sample(random, T);

    wfFilter.setNumParticles(N);
    wfFilter.setResampleOnly(false);

    log.info("rep\tt\tfilter.type\tmeasurement.type\tresample.type\tmeasurement");

    GaussianArHmmClassEvaluator wfClassEvaluator = new GaussianArHmmClassEvaluator("wf-pl", 
        null);
    GaussianArHmmRmseEvaluator wfRmseEvaluator = new GaussianArHmmRmseEvaluator("wf-pl", 
        null);
    GaussianArHmmPsiLearningEvaluator wfPsiEvaluator = new GaussianArHmmPsiLearningEvaluator("wf-pl", 
        truePsis, null);

    RingAccumulator<MutableDouble> wfLatency = 
        new RingAccumulator<MutableDouble>();
    Stopwatch wfWatch = new Stopwatch();


    for (int k = 0; k < K; k++) {
      log.info("Processing replication " + k);
      CountedDataDistribution<GaussianArHpTransitionState> wfDistribution =
          (CountedDataDistribution<GaussianArHpTransitionState>) wfFilter.getUpdater().createInitialParticles(N);


      final long numPreRuns = -1l;//wfDistribution.getMaxValueKey().getTime();
      
      /*
       * Recurse through the particle filter
       */
      for (int i = 0; i < T; i++) {
  
        final double x = simulation.get(i).getClassId();
        final Vector y = simulation.get(i).getObservedValue();

        if (i > numPreRuns) {

          if (i > 0) {
            wfWatch.reset();
            wfWatch.start();
            wfFilter.update(wfDistribution, simulation.get(i));
            wfWatch.stop();
            final long latency = wfWatch.elapsed(TimeUnit.MILLISECONDS);
            wfLatency.accumulate(new MutableDouble(latency));
            log.info(Joiner.on("\t").join(new String[] {
                Integer.toString(k), Integer.toString(i), 
                "wf-pl", "latency", "NA", 
                Long.toString(latency)})
                );
          }
          
          wfClassEvaluator.evaluate(k, simulation.get(i), wfDistribution);
          wfRmseEvaluator.evaluate(k, simulation.get(i), wfDistribution);
          wfPsiEvaluator.evaluate(k, simulation.get(i), wfDistribution);
        }

        if ((i+1) % (T/4d) < 1) {
          log.info("avg. wf latency=" + wfLatency.getMean().value);
          log.info("avg. wfRmse=" + wfRmseEvaluator.getTotalRate().getMean().value);
          log.info("avg. wfClassRate=" + wfClassEvaluator.getTotalRate().getMean().value);
          log.info("avg. wfPsi=" + wfPsiEvaluator.getTotalRate());
        }
      }

    }

    assertEquals(0d, wfRmseEvaluator.getTotalRate().getMean().value, 9e-1);
  }

}
