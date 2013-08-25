package hmm.categorical;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.statistics.DataDistribution;
import hmm.HmmTransitionState;
import hmm.HmmPlFilter;

import java.util.Random;

import org.apache.log4j.Logger;

import utils.ObservedValue;

import com.google.common.collect.Iterables;

/**
 * A particle filter for categorical response HMMs that provides the option of
 * water-filling resampling the expanded step-forward state space.
 * 
 * @author bwillard
 * 
 */
public class CategoricalHmmPlFilter extends HmmPlFilter<Integer> {

  public class CategoricalHMMPLUpdater extends HmmPlUpdater<Integer> {

    private static final long serialVersionUID = 7961478795131339665L;

    public CategoricalHMMPLUpdater(HiddenMarkovModel<Integer> hmm,
      Random rng) {
      super(hmm, rng);
    }

    @Override
    public double computeLogLikelihood(
      HmmTransitionState<Integer> particle,
      ObservedValue<Integer> observation) {
      final ComputableDistribution<Integer> f =
          Iterables.get(particle.getHmm().getEmissionFunctions(),
              particle.getState());
      return f.getProbabilityFunction().logEvaluate(
          observation.getObservedState());
    }

    @Override
    public HmmTransitionState<Integer> update(
      HmmTransitionState<Integer> previousParameter) {
      return previousParameter.clone();
    }

  }

  private static final long serialVersionUID = -7387680621521036135L;

  final Logger log = Logger.getLogger(CategoricalHmmPlFilter.class);

  public CategoricalHmmPlFilter(HiddenMarkovModel<Integer> hmm,
    Random rng, boolean resampleOnly) {
    super(resampleOnly);
    this.setUpdater(new CategoricalHMMPLUpdater(hmm, rng));
    this.setRandom(rng);
    this.resampleOnly = resampleOnly;
  }

  @Override
  public void update(
    DataDistribution<HmmTransitionState<Integer>> target,
    ObservedValue<Integer> data) {
    super.update(target, data);
  }

}
