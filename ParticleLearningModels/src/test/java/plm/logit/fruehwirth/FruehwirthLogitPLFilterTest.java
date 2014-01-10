package plm.logit.fruehwirth;

import static org.junit.Assert.fail;
import gov.sandia.cognition.math.MultivariateStatisticsUtil;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.RingAverager;
import gov.sandia.cognition.math.WeightedNumberAverager;
import gov.sandia.cognition.math.WeightedRingAverager;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.LogisticDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import junit.framework.Assert;

import org.junit.Test;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.statslibextensions.statistics.bayesian.DlmUtils;
import com.statslibextensions.util.ObservedValue;
import com.statslibextensions.util.ObservedValue.SimObservedValue;

public class FruehwirthLogitPLFilterTest {

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
    final int N = 3000;
    final List<ObservedValue<Vector, Matrix>> observations = Lists.newArrayList();

    List<SimObservedValue<Vector, Matrix, Vector>> dlmSamples = DlmUtils.sampleDlm(
        rng, N, trueInitialPrior, initialFilter);
    for (SimObservedValue<Vector, Matrix, Vector> samplePair : dlmSamples) {
      final double ev1Upper = -Math.log(-Math.log(rng.nextDouble()));
      final double upperUtility = samplePair.getObservedValue().getElement(0) +
          ev1Upper;
      final double lowerUtility = -Math.log(-Math.log(rng.nextDouble()));
      final double obs = (upperUtility > lowerUtility) ? 1d : 0d;
      observations.add(
          SimObservedValue.<Vector, Matrix, LogitTrueState>create(
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
        {1d}});

    final FruehwirthLogitPLFilter plFilter =
        new FruehwirthLogitPLFilter(initialPrior, 
            F, G, modelCovariance, false, rng);
    plFilter.setNumParticles(2000);

    final DataDistribution<FruehwirthLogitParticle> currentMixtureDistribution =
        plFilter.createInitialLearnedObject();
    double lastRMSE = Double.POSITIVE_INFINITY;
    for (int i = 0; i < N; i++) {
      final ObservedValue<Vector, Matrix> observation = observations.get(i);
      System.out.println("obs:" + observation);
      plFilter.update(currentMixtureDistribution, observation);

      List<WeightedValue<Vector>> wMeanValues = Lists.newArrayList();
      List<WeightedValue<Matrix>> wCovValues = Lists.newArrayList();
      final Vector trueState = dlmSamples.get(i).getTrueState();
      double sum = 0d;
      double sqSum = 0d;
      final double distTotalLogProb = currentMixtureDistribution.getTotal();
      for (Entry<FruehwirthLogitParticle, ? extends Number> particleEntry : 
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
      System.out.println("posterior RMSE: " + sum + " (" + sqSum + ")");
      Pair<Vector, Matrix> wMeanRes = MultivariateStatisticsUtil.computeWeightedMeanAndCovariance(wMeanValues);
      Pair<Vector, Matrix> wCovRes = MultivariateStatisticsUtil.computeWeightedMeanAndCovariance(wCovValues);
      System.out.println("\ttrue: " + trueState + '\n'
          + ", pMean:" + wMeanRes.getFirst()
          + " (" + wMeanRes.getSecond() + ")"
          + ", pCov:" + wCovRes.getFirst() 
          + " (" + wCovRes.getSecond() + ")"
          );
    }

    System.out.println("finished simulation");

    Assert.assertTrue(lastRMSE < 0.5d);

  }

  @Test
  public void test2() {

    final Random rng = new Random(829351983l);
    
    /*
     * Sample from a logit model.
     */
    final KalmanFilter initialFilter = new KalmanFilter(
          new LinearDynamicalSystem(
              MatrixFactory.getDefault().copyArray(new double[][] {
                  {1d, 0d},
                  {0d, 1d}}),
              MatrixFactory.getDefault().copyArray(new double[][] {
                  {0d, 0d},
                  {0d, 0d}}),
              MatrixFactory.getDefault().copyArray(new double[][] {
                  {1d, 1d}})),
          MatrixFactory.getDefault().copyArray(new double[][] {
              {0d, 0d},
              {0d, 0d}}),
          MatrixFactory.getDefault().copyArray(new double[][] {{0d}})    
        );
    
    final MultivariateGaussian trueInitialPrior = new MultivariateGaussian(
        VectorFactory.getDefault().copyValues(-0.7d, 0.9d),
        MatrixFactory.getDefault().copyArray(new double[][] {
            {1d, 0d},
            {0d, 1d}}));
    final int N = 100;
    final List<ObservedValue<Vector, Matrix>> observations = Lists.newArrayList();

    List<SimObservedValue<Vector, Matrix, Vector>> dlmSamples = DlmUtils.sampleDlm(
        rng, N, trueInitialPrior, initialFilter);
    for (SimObservedValue<Vector, Matrix, Vector> samplePair : dlmSamples) {
      final double ev1Upper = -Math.log(-Math.log(rng.nextDouble()));
      final double upperUtility = samplePair.getObservedValue().getElement(0) +
          ev1Upper;
      final double lowerUtility = -Math.log(-Math.log(rng.nextDouble()));
      final double obs = (upperUtility > lowerUtility) ? 1d : 0d;
      observations.add(
          SimObservedValue.<Vector, Matrix, LogitTrueState>create(
              VectorFactory.getDefault().copyValues(obs),
              samplePair.getObservedData(),
              new LogitTrueState(samplePair.getTrueState(), upperUtility, ev1Upper, lowerUtility)));
    }

    /*
     * Create and initialize the PL filter
     */
    final MultivariateGaussian initialPrior = new MultivariateGaussian(
        VectorFactory.getDefault().copyValues(0d, 0d),
        MatrixFactory.getDefault().copyArray(new double[][] {
            {10d, 0d},
            {0d, 10d}}));
    final Matrix F = MatrixFactory.getDefault().copyArray(new double[][] {
        {1d, 1d}});
    final Matrix G = MatrixFactory.getDefault().copyArray(new double[][] {
        {1d, 0d}, 
        {0d, 1d}});
    final Matrix modelCovariance = MatrixFactory.getDefault().copyArray(new double[][] {
        {0d, 0d},
        {0d, 0d}});

    final FruehwirthLogitPLFilter plFilter =
        new FruehwirthLogitPLFilter(initialPrior, 
            F, G, modelCovariance, true, rng);
    plFilter.setNumParticles(50);

    final DataDistribution<FruehwirthLogitParticle> currentMixtureDistribution =
        plFilter.createInitialLearnedObject();
    double lastRMSE = Double.POSITIVE_INFINITY;
    for (int i = 0; i < N; i++) {
      final ObservedValue<Vector, Matrix> observation = observations.get(i);
      System.out.println("obs:" + observation);
      plFilter.update(currentMixtureDistribution, observation);

      List<WeightedValue<Vector>> wMeanValues = Lists.newArrayList();
      List<WeightedValue<Matrix>> wCovValues = Lists.newArrayList();
      final Vector trueState = dlmSamples.get(i).getTrueState();
      double sum = 0d;
      double sqSum = 0d;
      final double distTotalLogProb = currentMixtureDistribution.getTotal();
      for (Entry<FruehwirthLogitParticle, ? extends Number> particleEntry : 
        currentMixtureDistribution.asMap().entrySet()) {
        final Vector particleState = particleEntry.getKey().getLinearState().getMean();
        final double rse = trueState.minus(particleState).norm2();
        final double weight = Math.exp(particleEntry.getValue().doubleValue() - distTotalLogProb);
        wMeanValues.add(DefaultWeightedValue.create(particleState, weight));
        wCovValues.add(DefaultWeightedValue.create(
            particleEntry.getKey().getLinearState().getCovariance(), weight));
        final double wRse = rse * weight;
        sum += wRse;
        sqSum += rse * wRse; 
      }
      lastRMSE = sum;
      System.out.println("posterior RMSE: " + sum + " (" + sqSum + ")");
      Pair<Vector, Matrix> wMeanRes = MultivariateStatisticsUtil.computeWeightedMeanAndCovariance(wMeanValues);
      Pair<Vector, Matrix> wCovRes = MultivariateStatisticsUtil.computeWeightedMeanAndCovariance(wCovValues);
      System.out.println("\ttrue: " + trueState + '\n'
          + ", pMean:" + wMeanRes.getFirst()
          + " (" + wMeanRes.getSecond() + ")"
          + ", pCov:" + wCovRes.getFirst() 
          + " (" + wCovRes.getSecond() + ")"
          );
    }

    System.out.println("finished simulation");

    Assert.assertTrue(lastRMSE < 1d);

  }


  /**
   * Tests a simple regression, i.e. z = a * x
   */
  @Test
  public void test3() {

    final Random rng = new Random(829351983l);
    
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
        VectorFactory.getDefault().copyValues(0.9d),
        MatrixFactory.getDefault().copyArray(new double[][] {
            {0d}}));
    final int N = 2000;
    final List<ObservedValue<Vector, Matrix>> observations = Lists.newArrayList();

    List<SimObservedValue<Vector, Matrix, Vector>> dlmSamples = DlmUtils.sampleDlm(
        rng, N, trueInitialPrior, initialFilter, 
        new Function<Matrix, Matrix>() {
          @Override
          public Matrix apply(Matrix input) {
            final double curVal;
            if (input == null)
              curVal = -1d;
            else
              curVal = input.getElement(0, 0);
            return MatrixFactory.getDefault().copyArray(
                new double[][] {{
                curVal + 2d/N
                }});
          }
        });
    for (SimObservedValue<Vector, Matrix, Vector> samplePair : dlmSamples) {
      final double ev1Upper = -Math.log(-Math.log(rng.nextDouble()));
      final double upperUtility = samplePair.getObservedValue().getElement(0) +
          ev1Upper;
      final double lowerUtility = -Math.log(-Math.log(rng.nextDouble()));
      final double obs = (upperUtility > lowerUtility) ? 1d : 0d;
      observations.add(
          SimObservedValue.<Vector, Matrix, LogitTrueState>create(
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
            {100d}}));
    final Matrix F = MatrixFactory.getDefault().copyArray(new double[][] {
        {1d}});
    final Matrix G = MatrixFactory.getDefault().copyArray(new double[][] {
        {1d}});
    final Matrix modelCovariance = MatrixFactory.getDefault().copyArray(new double[][] {
        {0d}});

    final FruehwirthLogitPLFilter plFilter =
        new FruehwirthLogitPLFilter(initialPrior, F, G, 
            modelCovariance, true, rng);
    plFilter.setNumParticles(1000);

    double lastRMSE = Double.POSITIVE_INFINITY;
    final DataDistribution<FruehwirthLogitParticle> currentMixtureDistribution =
        plFilter.createInitialLearnedObject();
    for (int i = 0; i < N; i++) {
      final ObservedValue<Vector, Matrix> observation = observations.get(i);
      System.out.println("obs:" + observation);
      plFilter.update(currentMixtureDistribution, observation);

      List<WeightedValue<Vector>> wMeanValues = Lists.newArrayList();
      List<WeightedValue<Matrix>> wCovValues = Lists.newArrayList();
      final Vector trueState = dlmSamples.get(i).getTrueState();
      double sum = 0d;
      double sqSum = 0d;
      final double distTotalLogProb = currentMixtureDistribution.getTotal();
      for (Entry<FruehwirthLogitParticle, ? extends Number> particleEntry : 
        currentMixtureDistribution.asMap().entrySet()) {
        final Vector particleState = particleEntry.getKey().getLinearState().getMean();
        final double rse = trueState.minus(particleState).norm2();
        final double weight = Math.exp(particleEntry.getValue().doubleValue() - distTotalLogProb);
        wMeanValues.add(DefaultWeightedValue.create(particleState, weight));
        wCovValues.add(DefaultWeightedValue.create(
            particleEntry.getKey().getLinearState().getCovariance(), weight));

        final double wRse = rse * weight;
        sum += wRse;
        sqSum += rse * wRse; 
      }
      lastRMSE = sum;
      System.out.println("posterior RMSE: " + sum + " (" + sqSum + ")");
      Pair<Vector, Matrix> wMeanRes = MultivariateStatisticsUtil.computeWeightedMeanAndCovariance(wMeanValues);
      Pair<Vector, Matrix> wCovRes = MultivariateStatisticsUtil.computeWeightedMeanAndCovariance(wCovValues);
      System.out.println("\ttrue: " + trueState + '\n'
          + ", pMean:" + wMeanRes.getFirst()
          + " (" + wMeanRes.getSecond() + ")"
          + ", pCov:" + wCovRes.getFirst() 
          + " (" + wCovRes.getSecond() + ")"
          );
    }

    System.out.println("finished simulation");

    Assert.assertTrue(lastRMSE < 1d);

  }

}
