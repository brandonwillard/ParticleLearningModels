package plm.gaussian;

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

import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.Test;

import plm.hmm.DlmHiddenMarkovModel;
import plm.hmm.GenericHMM.SimHmmObservedValue;
import plm.hmm.HmmPlFilter;
import plm.hmm.gaussian.GaussianArHpHmmPLFilter;
import plm.hmm.gaussian.GaussianArHpTransitionState;
import plm.util.gaussian.GaussianArHpEvaluator;
import plm.util.hmm.HmmResampleComparisonRunner;
import plm.util.hmm.gaussian.GaussianArHmmClassEvaluator;
import plm.util.hmm.gaussian.GaussianArHmmPsiLearningEvaluator;
import plm.util.hmm.gaussian.GaussianArHmmRmseEvaluator;

import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.statslibextensions.statistics.bayesian.DlmUtils;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;
import com.statslibextensions.util.ExtSamplingUtils;
import com.statslibextensions.util.ObservedValue.SimObservedValue;

public class GaussianArHpWfPlFilterTest {

  protected static final Logger log = Logger
        .getLogger(HmmResampleComparisonRunner.class);
  static {
    log.setLevel(Level.INFO);
  }

  @Test
  public void test() {
    final long seed = new Random().nextLong();
    final Random random = new Random(seed);
    log.info("seed=" + seed);
    ExtSamplingUtils.log.setLevel(Level.INFO);

    final double trueSigma2 = Math.pow(0.2d, 2);
    Matrix modelCovariance = MatrixFactory.getDefault().copyArray(
        new double[][] {{trueSigma2}});
    Matrix measurementCovariance = MatrixFactory.getDefault().copyArray(
        new double[][] {{trueSigma2}});

    Vector truePsi = VectorFactory.getDefault().copyValues(3d, 0.2d);

    LinearDynamicalSystem dlm = new LinearDynamicalSystem(
        MatrixFactory.getDefault().copyArray(new double[][] {{truePsi.getElement(1)}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}})
      );
    KalmanFilter trueKf = new KalmanFilter(dlm, modelCovariance, measurementCovariance);
    trueKf.setCurrentInput(VectorFactory.getDefault().copyValues(truePsi.getElement(0)));
    
    final double sigmaPriorMean = Math.pow(0.4, 2);
    final double sigmaPriorShape = 2d;
    final double sigmaPriorScale = sigmaPriorMean*(sigmaPriorShape - 1d);
    final InverseGammaDistribution sigmaPrior = new InverseGammaDistribution(sigmaPriorShape,
        sigmaPriorScale);
    
    final Vector phiMean = VectorFactory.getDefault().copyArray(new double[] {
        0d, 0.8d
    });
    final Matrix phiCov = MatrixFactory.getDefault().copyArray(new double[][] {
        {2d + 4d * sigmaPriorMean, 0d},
        { 0d, 4d * sigmaPriorMean}
    });
    final MultivariateGaussian phiPrior = new MultivariateGaussian(phiMean, phiCov);

    final int K = 3;
    final int T = 700;
    final int N = 1000;

    final GaussianArHpWfPlFilter wfFilter =
        new GaussianArHpWfPlFilter(trueKf, sigmaPrior, phiPrior, random, K, true);

    /*
     * Note: replications are over the same set of simulated observations.
     */
    List<SimObservedValue<Vector, Matrix, Vector>> simulations = DlmUtils.sampleDlm(
        random, T, trueKf.createInitialLearnedObject(), trueKf);

    wfFilter.setNumParticles(N);

//    log.info("rep\tt\tfilter.type\tmeasurement.type\tresample.type\tmeasurement");

    RingAccumulator<MutableDouble> wfLatency = 
        new RingAccumulator<MutableDouble>();
    Stopwatch wfWatch = new Stopwatch();

    RingAccumulator<MutableDouble> wfStateRMSEs = 
        new RingAccumulator<MutableDouble>();
    RingAccumulator<MutableDouble> wfPsiRMSEs = 
        new RingAccumulator<MutableDouble>();
    RingAccumulator<MutableDouble> wfSigma2RMSEs = 
        new RingAccumulator<MutableDouble>();

    for (int k = 0; k < K; k++) {
      log.info("Processing replication " + k);

      GaussianArHpEvaluator wfEvaluator = new GaussianArHpEvaluator("wf-pl", 
          truePsi, trueSigma2, null);
      CountedDataDistribution<GaussianArHpWfParticle> wfDistribution =
          (CountedDataDistribution<GaussianArHpWfParticle>) wfFilter.getUpdater().createInitialParticles(N);

      final long numPreRuns = -1l;//wfDistribution.getMaxValueKey().getTime();
      
      /*
       * Recurse through the particle filter
       */
      for (int i = 0; i < T; i++) {
  
        if (i > numPreRuns) {

          if (i > 0) {
            wfWatch.reset();
            wfWatch.start();
            wfFilter.update(wfDistribution, simulations.get(i));
            wfWatch.stop();
            final long latency = wfWatch.elapsed(TimeUnit.MILLISECONDS);
            wfLatency.accumulate(new MutableDouble(latency));
          }
          
          wfEvaluator.evaluate(k, simulations.get(i), wfDistribution);
        }

        if ((i+1) % (T/4d) < 1) {
          log.info(Joiner.on("\t").join(new String[] {
              Integer.toString(k), 
              Integer.toString(i), 
              Double.toString(wfLatency.getMean().value),
              Double.toString(wfEvaluator.getStateLastRmse()),
              Double.toString(wfEvaluator.getSigma2LastRmse()),
              Double.toString(wfEvaluator.getPsiLastRmse())
//              Double.toString(wfEvaluator.getRunningStateRmse()),
//              Double.toString(wfEvaluator.getRunningSigma2Rmse()),
//              Double.toString(wfEvaluator.getRunningPsiRmse())
              }));
          log.info(Joiner.on("\t").join(new String[] {
              Integer.toString(k), 
              Integer.toString(i), 
              Double.toString(wfLatency.getMean().value),
              "% " + Double.toString(wfEvaluator.getStateLastErrRate()),
              "% " + Double.toString(wfEvaluator.getSigma2LastErrRate()),
              "% " + Double.toString(wfEvaluator.getPsiLastErrRate())
//              Double.toString(wfEvaluator.getRunningStateErrRate()),
//              Double.toString(wfEvaluator.getRunningSigma2ErrRate()),
//              Double.toString(wfEvaluator.getRunningPsiErrRate())
              }));
        }
      }
      wfStateRMSEs.accumulate(new MutableDouble(wfEvaluator.getStateLastErrRate()));
      wfPsiRMSEs.accumulate(new MutableDouble(wfEvaluator.getPsiLastErrRate()));
      wfSigma2RMSEs.accumulate(new MutableDouble(wfEvaluator.getSigma2LastErrRate()));

    }

    assertEquals(0d, wfStateRMSEs.getMean().doubleValue(), 1d/Math.sqrt(N));
  }

}
