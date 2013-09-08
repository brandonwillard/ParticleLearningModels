package hmm.gaussian;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.bayesian.ParticleFilter;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;
import hmm.BasicHMM.SimHmmObservedValue;
import hmm.DlmHiddenMarkovModel;
import hmm.HmmPlFilter;
import hmm.HmmResampleComparisonRunner;
import hmm.HmmTransitionState;
import hmm.HmmTransitionState.ResampleType;

import java.io.FileWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import utils.CountedDataDistribution;
import utils.ObservedValue;
import utils.SamplingUtils;
import au.com.bytecode.opencsv.CSVWriter;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.math.DoubleMath;

public class GaussianArHmmRunner extends HmmResampleComparisonRunner {

  public static void main(String[] args) throws IOException {

    final long seed = new Random().nextLong();
    final Random random = new Random(seed);
    log.info("seed=" + seed);

    final int N = 10000;

    final double[] a = {0.9d, 0.9d};
    final double[] sigma2 = {Math.pow(0.2d, 2), Math.pow(1.2d, 2)};
    final double sigma_y2 = Math.pow(0.3d, 2);
    Matrix modelCovariance1 = MatrixFactory.getDefault().copyArray(new double[][] {{0.2d}});
    Matrix modelCovariance2 = MatrixFactory.getDefault().copyArray(new double[][] {{1.2d}});
    Matrix measurementCovariance = MatrixFactory.getDefault().copyArray(new double[][] {{0.3d}});
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

    final HiddenMarkovModel<Double> trueHmm1 =
        new HiddenMarkovModel<Double>(initialClassProbs,
            classTransProbs, Lists.newArrayList(
                s1Likelihood, s2Likelihood));

    final HmmPlFilter<Double> wfFilter =
        new GaussianArHmmPlFilter(trueHmm1, prior, a, sigma2, sigma_y2, random, true);
    final GaussianArHmmMkfFilter rsFilter =
        new GaussianArHmmMkfFilter(trueHmm1, prior, a, sigma2, sigma_y2, random);


    String outputFilename = args[0] + "/hmm-nar-wf-rs-10000-class-errors-m1.csv";
    final int K = 10;
    final int T = 300;

    List<SimHmmObservedValue<Vector, Vector>> simulation = dlmHmm.sample(random, T);

    wfFilter.setNumParticles(N);
    wfFilter.setResampleOnly(false);
    rsFilter.setNumParticles(N);

    CSVWriter writer = new CSVWriter(new FileWriter(outputFilename), ',');
    String[] header = "rep,t,filter.type,measurement.type,resample.type,measurement".split(",");
    writer.writeNext(header);

    GaussianArHmmClassEvaluator wfClassEvaluator = new GaussianArHmmClassEvaluator("wf-pl", 
        writer);
    GaussianArHmmClassEvaluator mkfClassEvaluator = new GaussianArHmmClassEvaluator("mkf", 
        writer);
    GaussianArHmmRmseEvaluator wfRmseEvaluator = new GaussianArHmmRmseEvaluator("wf-pl", 
        writer);
    GaussianArHmmRmseEvaluator mkfRmseEvaluator = new GaussianArHmmRmseEvaluator("mkf", 
        writer);

    RingAccumulator<MutableDouble> mkfLatency = 
        new RingAccumulator<MutableDouble>();
    RingAccumulator<MutableDouble> wfLatency = 
        new RingAccumulator<MutableDouble>();
    Stopwatch mkfWatch = new Stopwatch();
    Stopwatch wfWatch = new Stopwatch();


    for (int k = 0; k < K; k++) {
      log.info("Processing replication " + k);
      CountedDataDistribution<HmmTransitionState<Double>> wfDistribution =
          (CountedDataDistribution<HmmTransitionState<Double>>) wfFilter.getUpdater().createInitialParticles(N);
//          ((HmmPlFilter.HmmPlUpdater<Double>) wfFilter.getUpdater()).baumWelchInitialization(sample.getFirst(), N);

      DataDistribution<GaussianArTransitionState> rsDistribution =
          rsFilter.getUpdater().createInitialParticles(N);

      final long numPreRuns = -1l;//wfDistribution.getMaxValueKey().getTime();
      
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
          mkfWatch.reset();
          mkfWatch.start();
          rsFilter.update(rsDistribution, obsState);
          mkfWatch.stop();
          final long latency = mkfWatch.elapsed(TimeUnit.MILLISECONDS);
          mkfLatency.accumulate(new MutableDouble(latency));
          writer.writeNext(new String[] {
              Integer.toString(k), Integer.toString(i), 
              "mkf", "latency", "NA", 
              Long.toString(latency)
          });
        }

        if (i > numPreRuns) {

          if (i > 0) {
            wfWatch.reset();
            wfWatch.start();
            wfFilter.update(wfDistribution, obsState);
            wfWatch.stop();
            final long latency = wfWatch.elapsed(TimeUnit.MILLISECONDS);
            wfLatency.accumulate(new MutableDouble(latency));
            writer.writeNext(new String[] {
                Integer.toString(k), Integer.toString(i), 
                "wf-pl", "latency", "NA", 
                Long.toString(latency)
            });
          }
          
          wfClassEvaluator.evaluate(k, simulation.get(i), wfDistribution);
          mkfClassEvaluator.evaluate(k, simulation.get(i), rsDistribution);
          wfRmseEvaluator.evaluate(k, simulation.get(i), wfDistribution);
          mkfRmseEvaluator.evaluate(k, simulation.get(i), rsDistribution);
        }
      }
    }

    log.info("avg. mkf latency=" + mkfLatency.getMean().value);
    log.info("avg. wf latency=" + wfLatency.getMean().value);
    log.info("avg. mkfRmse=" + mkfRmseEvaluator.getTotalRate().getMean().value);
    log.info("avg. wfRmse=" + wfRmseEvaluator.getTotalRate().getMean().value);
    log.info("avg. mkfClassRate=" + mkfClassEvaluator.getTotalRate().getMean().value);
    log.info("avg. wfClassRate=" + wfClassEvaluator.getTotalRate().getMean().value);
    writer.close();
  }
    

}
