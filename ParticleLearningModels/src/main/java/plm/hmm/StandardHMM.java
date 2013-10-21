package plm.hmm;

import gov.sandia.cognition.collection.CollectionUtil;
import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.DefaultPair;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

/**
 * HMM without state dependencies.
 * @author bwillar0
 *
 * @param <ResponseType>
 */
public class StandardHMM<ResponseType> extends AbstractCloneableSerializable implements GenericHMM<ResponseType, Integer, DataDistribution<Integer>> {
  
  protected ExposedHmm<ResponseType> hmm;

  public StandardHMM(Vector initialProbs, Matrix transProbs, List<? extends ComputableDistribution<ResponseType>> emissions) {
    hmm = new ExposedHmm<ResponseType>(new HiddenMarkovModel<>(initialProbs, transProbs, emissions));
  }

  @Override
  public List<SimHmmObservedValue<ResponseType, Integer>> sample(
      Random random, int numSamples) {

    List<SimHmmObservedValue<ResponseType, Integer>> results = Lists.newArrayList();
    Vector p = hmm.getInitialProbability();
    int state = -1;
    for (int n = 0; n < numSamples; n++) {
      double value = random.nextDouble();
      state = -1;
      while (value > 0.0) {
        state++;
        value -= p.getElement(state);
      }
  
      ResponseType sample =
          CollectionUtil.getElement(hmm.getEmissionFunctions(),
              state).sample(random);
      results.add(new SimHmmObservedValue<ResponseType, Integer>(n, state, state, sample));
      p = hmm.getTransitionProbability().getColumn(state);
    }
  
    return results;
  }

  @Override
  public int getNumStates() {
    return hmm.getNumStates();
  }

  @Override
  public Matrix getTransitionProbability() {
    return hmm.getTransitionProbability();
  }

  public ArrayList<Integer> viterbi(
      Collection<? extends ResponseType> observations) {
    return hmm.viterbi(observations);
  }

  public ArrayList<Vector> stateBeliefs(
      Collection<? extends ResponseType> observations) {
    return hmm.stateBeliefs(observations);
  }

  protected static class ExposedHmm<ObservationType> extends HiddenMarkovModel<ObservationType> {
  
    private static final long serialVersionUID = 7707497613044304004L;
  
    public ExposedHmm(HiddenMarkovModel<ObservationType> hmm) {
      this.emissionFunctions = hmm.getEmissionFunctions();
      this.initialProbability = hmm.getInitialProbability();
      this.transitionProbability = hmm.getTransitionProbability();
    }
  
    @Override
    protected WeightedValue<Vector> computeForwardProbabilities(
      Vector alpha, Vector b, boolean normalize) {
      return super.computeForwardProbabilities(alpha, b, normalize);
    }
  
    @Override
    protected void computeObservationLikelihoods(
      ObservationType observation, Vector b) {
      super.computeObservationLikelihoods(observation, b);
    }
  
    @Override
    protected WeightedValue<Vector> computeBackwardProbabilities(
      Vector beta, Vector b, double weight) {
      return super.computeBackwardProbabilities(beta, b, weight);
    }
  
    @Override
    public ArrayList<Vector> computeStateObservationLikelihood(
      ArrayList<WeightedValue<Vector>> alphas,
      ArrayList<WeightedValue<Vector>> betas, double scaleFactor) {
      return super.computeStateObservationLikelihood(alphas, betas,
          scaleFactor);
    }
  
    @Override
    protected Matrix computeTransitions(
      ArrayList<WeightedValue<Vector>> alphas,
      ArrayList<WeightedValue<Vector>> betas, ArrayList<Vector> b) {
      return super.computeTransitions(alphas, betas, b);
    }
  
    @Override
    protected WeightedValue<Integer> findMostLikelyState(
      int destinationState, Vector delta) {
      return super.findMostLikelyState(destinationState, delta);
    }
  
    @Override
    protected Pair<Vector, int[]> computeViterbiRecursion(Vector delta,
      Vector bn) {
      return super.computeViterbiRecursion(delta, bn);
    }
  
    @Override
    public  
        double
        computeMultipleObservationLogLikelihood(
          Collection<? extends Collection<? extends ObservationType>> sequences) {
      return super.computeMultipleObservationLogLikelihood(sequences);
    }
  
    @Override
    public ArrayList<WeightedValue<Vector>>
        computeForwardProbabilities(ArrayList<Vector> b,
          boolean normalize) {
      return super.computeForwardProbabilities(b, normalize);
    }
  
    @Override
    public ArrayList<Vector> computeObservationLikelihoods(
      Collection<? extends ObservationType> observations) {
      return super.computeObservationLikelihoods(observations);
    }
  
    @Override
    public ArrayList<WeightedValue<Vector>>
        computeBackwardProbabilities(ArrayList<Vector> b,
          ArrayList<WeightedValue<Vector>> alphas) {
      return super.computeBackwardProbabilities(b, alphas);
    }
  }

  public ArrayList<WeightedValue<Vector>> computeForwardProbabilities(
      ArrayList<Vector> b, boolean normalize) {
    return hmm.computeForwardProbabilities(b, normalize);
  }

  public ArrayList<Vector> computeObservationLikelihoods(
      Collection<? extends ResponseType> observations) {
    return hmm.computeObservationLikelihoods(observations);
  }

  public ArrayList<WeightedValue<Vector>> computeBackwardProbabilities(
      ArrayList<Vector> b, ArrayList<WeightedValue<Vector>> alphas) {
    return hmm.computeBackwardProbabilities(b, alphas);
  }

  public ArrayList<Vector> computeStateObservationLikelihood(
      ArrayList<WeightedValue<Vector>> alphas,
      ArrayList<WeightedValue<Vector>> betas, double scaleFactor) {
    return hmm.computeStateObservationLikelihood(alphas, betas, scaleFactor);
  }

  @Override
  public DataDistribution<Integer> getEmissionFunction(DataDistribution<Integer> o, int classId) {
    return (DataDistribution<Integer>) Iterables.get(this.hmm.getEmissionFunctions(), classId);
  }

  @Override
  public Vector getClassMarginalProbabilities() {
    return this.hmm.getInitialProbability();
  }

  public static <T> StandardHMM<T> create(
      HiddenMarkovModel<T> hmm) {
    return new StandardHMM(hmm.getInitialProbability(), 
        hmm.getTransitionProbability(), (List) hmm.getEmissionFunctions());
  }

  @Override
  public StandardHMM<ResponseType> clone() {
    StandardHMM<ResponseType> clone = (StandardHMM<ResponseType>) super.clone();
    clone.hmm = this.hmm;
    return clone;
  }
}
