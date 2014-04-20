package plm.logit.fruehwirth;

import gov.sandia.cognition.math.MultivariateStatisticsUtil;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import junit.framework.Assert;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.Test;

import plm.util.logit.fruehwirth.LogitTrueState;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.statslibextensions.statistics.bayesian.DlmUtils;
import com.statslibextensions.util.ObservedValue;
import com.statslibextensions.util.ObservedValue.SimObservedValue;

public class LogitParRBCWFFilterTest {

  protected static final Logger log = Logger
      .getLogger(LogitParRBCWFFilterTest.class);

  static {
    log.setLevel(Level.INFO);
  }

  @Test
  public void test1() {

    final Random rng = new Random();//829351983l);
    
    /*
     * Sample from a logit model.
     */
    final KalmanFilter initialFilter = new KalmanFilter(
          new LinearDynamicalSystem(
              MatrixFactory.getDefault().copyArray(new double[][] {
                  {1d}}),
              MatrixFactory.getDefault().copyArray(new double[][] {
                  {0d}}),
              MatrixFactory.getDefault().copyArray(new double[][] {
                  {1d}})),
          MatrixFactory.getDefault().copyArray(new double[][] {
              {0d}}),
          MatrixFactory.getDefault().copyArray(new double[][] {{0d}})    
        );
    
    final MultivariateGaussian trueInitialPrior = new MultivariateGaussian(
        VectorFactory.getDefault().copyValues(3.7d),
        MatrixFactory.getDefault().copyArray(new double[][] {
            {0d}}));
    final int T = 3000;
    final List<ObservedValue<Vector, Matrix>> observations = Lists.newArrayList();

    List<SimObservedValue<Vector, Matrix, Vector>> dlmSamples = DlmUtils.sampleDlm(
        rng, T, trueInitialPrior, initialFilter);
    int t = 0;
    for (SimObservedValue<Vector, Matrix, Vector> samplePair : dlmSamples) {
      final double ev1Upper = -Math.log(-Math.log(rng.nextDouble()));
      final double upperUtility = samplePair.getObservedValue().getElement(0) +
          ev1Upper;
      final double lowerUtility = -Math.log(-Math.log(rng.nextDouble()));
      final double obs = (upperUtility > lowerUtility) ? 1d : 0d;
      observations.add(
          SimObservedValue.<Vector, Matrix, LogitTrueState>create(
              t++,
              VectorFactory.getDefault().copyValues(obs),
              samplePair.getObservedData(),
              new LogitTrueState(samplePair.getTrueState(), upperUtility, ev1Upper, lowerUtility)));
    }

    /*
     * Create and initialize the PL filter
     */
    final MultivariateGaussian initialPrior = new MultivariateGaussian(
        VectorFactory.getDefault().copyValues(0d),
        MatrixFactory.getDefault().copyArray(new double[][] {
            {1000d}}));
    final Matrix F = MatrixFactory.getDefault().copyArray(new double[][] {
        {1d}});
    final Matrix G = MatrixFactory.getDefault().copyArray(new double[][] {
        {1d}});
    final Matrix modelCovariance = MatrixFactory.getDefault().copyArray(new double[][] {
        {0d}});

    final LogitParRBCWFFilter plFilter =
        new LogitParRBCWFFilter(initialPrior, 
            F, G, modelCovariance, rng);
    plFilter.setNumParticles(4000);

    final DataDistribution<LogitMixParticle> currentMixtureDistribution =
        plFilter.createInitialLearnedObject();
    double lastRMSE = Double.POSITIVE_INFINITY;

    /*
     * Let's track latency...
     */
    Stopwatch watch = new Stopwatch();
    UnivariateGaussian.SufficientStatistic latencyStats = 
        new UnivariateGaussian.SufficientStatistic();

    for (int i = 0; i < T; i++) {
      final ObservedValue<Vector, Matrix> observation = observations.get(i);

      watch.reset();
      watch.start();
        plFilter.update(currentMixtureDistribution, observation);
      watch.stop();
      final long latency = watch.elapsed(TimeUnit.MILLISECONDS);
      latencyStats.update(latency);

      List<WeightedValue<Vector>> wMeanValues = Lists.newArrayList();
      List<WeightedValue<Matrix>> wCovValues = Lists.newArrayList();
      final Vector trueState = dlmSamples.get(i).getTrueState();
      double sum = 0d;
      double sqSum = 0d;
      final double distTotalLogProb = currentMixtureDistribution.getTotal();
      for (Entry<LogitMixParticle, ? extends Number> particleEntry : 
        currentMixtureDistribution.asMap().entrySet()) {
        final Vector particleState = particleEntry.getKey().getLinearState().getMean();
        final double rse = trueState.minus(particleState).dotDivide(trueState).norm2();
        final double weight = Math.exp(particleEntry.getValue().doubleValue() - distTotalLogProb);
        wMeanValues.add(DefaultWeightedValue.create(particleState, weight));
        wCovValues.add(DefaultWeightedValue.create(
            particleEntry.getKey().getLinearState().getCovariance(), weight));
        final double wRse = rse * weight;
        sum += wRse;
        sqSum += rse * wRse; 
      }
      lastRMSE = sum;

      if ((i+1) % (T/20d) < 1) {
        log.info("t = " + Integer.toString(i));
        log.info("update latency (ms) = " 
          + Double.toString(latencyStats.getMean()) + "("
          + Double.toString(latencyStats.getVariance()) + ")"
          );
        log.info("post. mean RMSE: " + sum + " (" + sqSum + ")");
        Pair<Vector, Matrix> wMeanRes = MultivariateStatisticsUtil.computeWeightedMeanAndCovariance(wMeanValues);
        Pair<Vector, Matrix> wCovRes = MultivariateStatisticsUtil.computeWeightedMeanAndCovariance(wCovValues);
        log.info("\ttrue: " + trueState + '\n'
            + ", pMean:" + wMeanRes.getFirst()
            + " (" + wMeanRes.getSecond().toString().trim() + ")"
            + ", pCov:" + wCovRes.getFirst() 
            + " (" + wCovRes.getSecond().toString().trim() + ")"
            );
      }
    }

    log.info("finished simulation");

    Assert.assertTrue(lastRMSE < 0.5d);

  }

}
