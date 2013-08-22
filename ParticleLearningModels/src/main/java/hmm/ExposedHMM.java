package hmm;

import java.util.ArrayList;
import java.util.Collection;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

public class ExposedHMM<ObservationType> extends HiddenMarkovModel<ObservationType> {

  private static final long serialVersionUID = 7707497613044304004L;

  public ExposedHMM(HiddenMarkovModel<ObservationType> hmm) {
    this.emissionFunctions = hmm.getEmissionFunctions();
    this.initialProbability = hmm.getInitialProbability();
    this.transitionProbability = hmm.getTransitionProbability();
  }

  @Override
  protected WeightedValue<Vector> computeForwardProbabilities(
    Vector alpha, Vector b, boolean normalize) {
    // TODO Auto-generated method stub
    return super.computeForwardProbabilities(alpha, b, normalize);
  }

  @Override
  protected void computeObservationLikelihoods(
    ObservationType observation, Vector b) {
    // TODO Auto-generated method stub
    super.computeObservationLikelihoods(observation, b);
  }

  @Override
  protected WeightedValue<Vector> computeBackwardProbabilities(
    Vector beta, Vector b, double weight) {
    // TODO Auto-generated method stub
    return super.computeBackwardProbabilities(beta, b, weight);
  }

  @Override
  protected ArrayList<Vector> computeStateObservationLikelihood(
    ArrayList<WeightedValue<Vector>> alphas,
    ArrayList<WeightedValue<Vector>> betas, double scaleFactor) {
    // TODO Auto-generated method stub
    return super.computeStateObservationLikelihood(alphas, betas,
        scaleFactor);
  }

  @Override
  protected Matrix computeTransitions(
    ArrayList<WeightedValue<Vector>> alphas,
    ArrayList<WeightedValue<Vector>> betas, ArrayList<Vector> b) {
    // TODO Auto-generated method stub
    return super.computeTransitions(alphas, betas, b);
  }

  @Override
  protected WeightedValue<Integer> findMostLikelyState(
    int destinationState, Vector delta) {
    // TODO Auto-generated method stub
    return super.findMostLikelyState(destinationState, delta);
  }

  @Override
  protected Pair<Vector, int[]> computeViterbiRecursion(Vector delta,
    Vector bn) {
    // TODO Auto-generated method stub
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
