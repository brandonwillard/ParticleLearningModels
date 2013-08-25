package hmm.gaussian;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.statistics.ComputableDistribution;
import hmm.HmmTransitionState;
import hmm.HmmPlFilter;

import java.util.Random;

import org.apache.log4j.Logger;

import com.google.common.collect.Iterables;

import utils.ObservedValue;

/**
 * A Particle Learning filter for a multivariate Gaussian Dirichlet Process.
 * 
 * @author bwillard
 * 
 */
public class GaussianHmmPlFilter extends HmmPlFilter<Double> {

  public class GaussianHMMPLUpdater extends HmmPlUpdater<Double> {

    final private HiddenMarkovModel<Double> hmm;
    final private Random rng;

    public GaussianHMMPLUpdater(HiddenMarkovModel<Double> hmm,
      Random rng) {
      super(hmm, rng);
      this.hmm = hmm;
      this.rng = rng;
    }

    @Override
    public double computeLogLikelihood(
      HmmTransitionState<Double> particle,
      ObservedValue<Double> observation) {
      final ComputableDistribution<Double> f =
          Iterables.get(particle.getHmm().getEmissionFunctions(),
              particle.getState());
      return f.getProbabilityFunction().logEvaluate(
          observation.getObservedState());
    }

    @Override
    public HmmTransitionState<Double> update(
      HmmTransitionState<Double> previousParameter) {
      return previousParameter.clone();
    }

  }

  final Logger log = Logger.getLogger(GaussianHmmPlFilter.class);

  public GaussianHmmPlFilter(HiddenMarkovModel<Double> hmm,
    Random rng, boolean resampleOnly) {
    super(resampleOnly);
    this.setUpdater(new GaussianHMMPLUpdater(hmm, rng));
    this.setRandom(rng);
  }

}
