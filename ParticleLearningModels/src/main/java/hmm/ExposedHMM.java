package hmm;

import java.util.ArrayList;
import java.util.Collection;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.util.WeightedValue;

public class ExposedHMM<ObservationType> extends HiddenMarkovModel<ObservationType> {

  private static final long serialVersionUID = 7707497613044304004L;

  public ExposedHMM(HiddenMarkovModel<ObservationType> hmm) {
    this.emissionFunctions = hmm.getEmissionFunctions();
    this.initialProbability = hmm.getInitialProbability();
    this.transitionProbability = hmm.getTransitionProbability();
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
