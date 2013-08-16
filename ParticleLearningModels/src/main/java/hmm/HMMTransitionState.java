package hmm;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Vector;

public class HMMTransitionState {
  
  final private HiddenMarkovModel<Double> hmm;
  private Vector state;

  public HiddenMarkovModel<Double> getHmm() {
    return hmm;
  }

  public HMMTransitionState(HiddenMarkovModel<Double> hmm, Vector state) {
    super();
    this.hmm = hmm;
  }

}
