package plm.hmm.gaussian;

import com.statslibextensions.util.ObservedValue;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import plm.hmm.DlmHiddenMarkovModel;
import plm.hmm.HmmTransitionState;

public class DlmTransitionState extends HmmTransitionState<Vector, DlmHiddenMarkovModel> {
  
  protected MultivariateGaussian state;

  public DlmTransitionState(
      DlmTransitionState prevState,
      DlmHiddenMarkovModel hmm, Integer classId,
      ObservedValue<Vector> data, MultivariateGaussian state) {
    super(prevState, hmm, classId, data);
    this.state = state;
  }

  public DlmTransitionState(DlmHiddenMarkovModel hmm, Integer classId,
      ObservedValue<Vector> data, MultivariateGaussian state) {
    super(hmm, classId, data);
    this.state = state;
  }

  @Override
  public DlmTransitionState clone() {
    DlmTransitionState clone = (DlmTransitionState) super.clone();
    clone.state = this.state.clone();
    return clone;
  }
  
  public MultivariateGaussian getState() {
    return this.state;
  }
}
