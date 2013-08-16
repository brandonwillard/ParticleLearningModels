package hmm;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.AbstractParticleFilter;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.MultivariateStudentTDistribution;
import gov.sandia.cognition.statistics.distribution.NormalInverseWishartDistribution;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.ObjectUtil;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import utils.LogMath2;
import utils.SamplingUtils;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

/**
 * A Particle Learning filter for a multivariate Gaussian Dirichlet Process.
 * 
 * @author bwillard
 * 
 */
public class CategoricalHMMPLFilter extends AbstractParticleFilter<Double, HMMTransitionState> {

  public class CategoricalHMMPLUpdater extends AbstractCloneableSerializable
      implements
        Updater<Double, HMMTransitionState> {

    final private HiddenMarkovModel<Double> hmm; 
    final private Random rng;

    public CategoricalHMMPLUpdater(HiddenMarkovModel<Double> hmm, Random rng) {
      this.hmm = hmm;
      this.rng = rng;
    }

    /**
     * In the case of a Particle Learning model, such as this, the prior predictive log likelihood
     * is used.
     */
    @Override
    public double computeLogLikelihood(HMMTransitionState particle, Double observation) {

      /*
       * Evaluate the log likelihood for a new component.
       * TODO
       */

      /*
       * Now, evaluate log likelihood for the current mixture components
       * TODO
       */

      return 0d;
    }

    @Override
    public DataDistribution<HMMTransitionState> createInitialParticles(int numParticles) {

      /**
       * TODO how to initialize?  FFBS?
       */
      final DefaultDataDistribution<HMMTransitionState> initialParticles =
          new DefaultDataDistribution<>(numParticles);
      for (int i = 0; i < numParticles; i++) {
        final HMMTransitionState particleMvgDPDist =
            new HMMTransitionState(this.hmm, null); 
        initialParticles.increment(particleMvgDPDist);
      }
      return initialParticles;
    }

    /**
     * In this model/filter, there's no need for blind samples from the predictive distribution.
     */
    @Override
    public HMMTransitionState update(HMMTransitionState previousParameter) {
      return previousParameter;
    }

  }

  public CategoricalHMMPLFilter(HiddenMarkovModel<Double> hmm, Random rng) {
    this.setUpdater(new CategoricalHMMPLUpdater(hmm, rng));
    this.setRandom(rng);
  }

  @Override
  public void update(DataDistribution<HMMTransitionState> target, Double data) {
    Preconditions.checkState(target.getDomainSize() == this.numParticles);

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final double[] cumulativeLogLikelihoods = new double[this.numParticles];
    final List<HMMTransitionState> particleSupport = Lists.newArrayList(target.getDomain());
    int j = 0;
    for (final HMMTransitionState particle : particleSupport) {
      final double logLikelihood = this.updater.computeLogLikelihood(particle, data);
      cumulativeLogLikelihoods[j] =
          j > 0 ? LogMath2.add(cumulativeLogLikelihoods[j - 1], logLikelihood) : logLikelihood;
      particleTotalLogLikelihood = LogMath2.add(particleTotalLogLikelihood, logLikelihood);
      j++;
    }

    final List<HMMTransitionState> resampledParticles =
        SamplingUtils.sampleMultipleLogScale(cumulativeLogLikelihoods, particleTotalLogLikelihood,
            particleSupport, random, this.numParticles);

    /*
     * Propagate
     */
    final DataDistribution<HMMTransitionState> updatedDist = new DefaultDataDistribution<>();
    for (final HMMTransitionState particle : resampledParticles) {

      /*
       * First, sample a mixture component index
       * TODO
       */
    }


    target.clear();
    target.incrementAll(updatedDist);
  }
}
