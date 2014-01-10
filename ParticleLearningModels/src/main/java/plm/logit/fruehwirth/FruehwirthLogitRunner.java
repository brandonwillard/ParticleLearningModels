package plm.logit.fruehwirth;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.LogisticDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.Pair;

import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import com.google.common.collect.Lists;
import com.statslibextensions.statistics.bayesian.DlmUtils;
import com.statslibextensions.util.ObservedValue;
import com.statslibextensions.util.ObservedValue.SimObservedValue;

/**
 * @author bwillard
 * 
 */
public class FruehwirthLogitRunner {
  
  public static void main(String[] args) {


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
                  {2d, 1d}})),
          MatrixFactory.getDefault().copyArray(new double[][] {
              {0d, 0d},
              {0d, 0d}}),
          MatrixFactory.getDefault().copyArray(new double[][] {{0d}})    
        );
    
    final MultivariateGaussian trueInitialPrior = new MultivariateGaussian(
        VectorFactory.getDefault().copyValues(0.5d, 3d),
        MatrixFactory.getDefault().copyArray(new double[][] {
            {1d, 0d},
            {0d, 1d}}));
    final int N = 100;
    final List<ObservedValue<Vector, Matrix>> observations = Lists.newArrayList();
    final LogisticDistribution ev1Dist = new LogisticDistribution(0d, 1d);

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
        new FruehwirthLogitPLFilter(initialPrior, F, G, modelCovariance, true, rng);
    plFilter.setNumParticles(10000);

    final DataDistribution<FruehwirthLogitParticle> currentMixtureDistribution =
        plFilter.createInitialLearnedObject();
    for (int i = 0; i < N; i++) {
      final ObservedValue<Vector, Matrix> observation = observations.get(i);
      System.out.println("obs:" + observation);
      plFilter.update(currentMixtureDistribution, observation);

      /*
       * Compute some summary stats. TODO We need to compute something informative for this
       * situation.
       */
      double rmseMean = 0d;

      for (Entry<FruehwirthLogitParticle, ? extends Number> particleEntry : 
        currentMixtureDistribution.asMap().entrySet()) {
        final Vector trueState = dlmSamples.get(i).getTrueState();
        final Vector particleState = particleEntry.getKey().getLinearState().getMean();
        final double rse = Math.sqrt(trueState.minus(particleState).norm2());
//        System.out.println("true: " + trueState + ", particle:" + particleState);

        rmseMean += rse * Math.exp(particleEntry.getValue().doubleValue());
      }
      System.out.println("posterior RMSE mean: " + rmseMean);
//          + " (" + rmseVar + ")");
    }

    System.out.println("finished simulation");

  }

}
