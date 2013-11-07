package plm.logit.fruehwirth;

import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.ExponentialDistribution;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.LogisticDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.MultivariateMixtureDensityModel;
import gov.sandia.cognition.statistics.distribution.NormalInverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.DefaultPair;
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
  
  public static class LogitTrueState {
    protected Vector state;
    protected double logitSample;

    public LogitTrueState(Vector state, double logitSample) {
      this.state = state;
      this.logitSample = logitSample;
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append("LogitTrueState [state=").append(this.state)
          .append(", logitSample=").append(this.logitSample).append("]");
      return builder.toString();
    }
    
  }

  public static void main(String[] args) {


    final Random rng = new Random(829351983l);
    
    /*
     * Sample from a logit model.
     */
    final KalmanFilter initialFilter = new KalmanFilter(
          new LinearDynamicalSystem(
              MatrixFactory.getDefault().copyArray(new double[][] {{1}}),
              MatrixFactory.getDefault().copyArray(new double[][] {{0}}),
              MatrixFactory.getDefault().copyArray(new double[][] {{1}})),
          MatrixFactory.getDefault().copyArray(new double[][] {{1}}),
          MatrixFactory.getDefault().copyArray(new double[][] {{0}})    
        );
    
    final MultivariateGaussian trueInitialPrior = new MultivariateGaussian(
        VectorFactory.getDefault().copyValues(0d),
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}}));
    final int N = 100;
    final List<ObservedValue<Vector, Matrix>> observations = Lists.newArrayList();
    final LogisticDistribution ev1Dist = new LogisticDistribution(0d, 1d);

    List<Pair<Vector, Vector>> dlmSamples = DlmUtils.sampleDlm(rng, N, trueInitialPrior, initialFilter);
    for (Pair<Vector, Vector> samplePair : dlmSamples) {
      final double logitSample = ev1Dist.sample(rng);
      final double augObs = samplePair.getFirst().getElement(0) + logitSample;
      final double obs = (augObs > 0d) ? 1d : 0d;
      observations.add(
          SimObservedValue.<Vector, Matrix, LogitTrueState>create(
              VectorFactory.getDefault().copyValues(obs),
              MatrixFactory.getDefault().copyArray(new double[][] {{1d}}),
              new LogitTrueState(samplePair.getSecond(), logitSample)));
    }

    /*
     * Create and initialize the PL filter
     */
    final MultivariateGaussian initialPrior = new MultivariateGaussian(
        VectorFactory.getDefault().copyValues(0d),
        MatrixFactory.getDefault().copyArray(new double[][] {{0.5d}}));
    final Matrix F = MatrixFactory.getDefault().copyArray(new double[][] {{1}});
    final Matrix G = MatrixFactory.getDefault().copyArray(new double[][] {{1}});
    final Matrix modelCovariance = MatrixFactory.getDefault().copyArray(new double[][] {{1}});
    final FruehwirthLogitPLFilter plFilter =
        new FruehwirthLogitPLFilter(initialPrior, F, G, modelCovariance, rng);
    plFilter.setNumParticles(50);

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
      final UnivariateGaussian.SufficientStatistic rmseSuffStat =
          new UnivariateGaussian.SufficientStatistic();

      for (Entry<FruehwirthLogitParticle, ? extends Number> particleEntry : 
        currentMixtureDistribution.asMap().entrySet()) {
        final Vector trueState = dlmSamples.get(i).getSecond();
        final Vector particleState = particleEntry.getKey().getLinearState().getMean();
        final double rse = Math.sqrt(trueState.minus(particleState).norm2());
        System.out.println("true: " + trueState + ", particle:" + particleState);

        rmseSuffStat.update(rse * Math.exp(particleEntry.getValue().doubleValue()));
      }
      System.out.println("posterior RMSE: " + rmseSuffStat.getMean()
          + " (" + rmseSuffStat.getVariance() + ")");
    }

    System.out.println("finished simulation");

  }

}
