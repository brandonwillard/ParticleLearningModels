package hmm;

import java.util.List;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.CloneableSerializable;
import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.WeightedValue;

/**
 * A simple class for tracking hidden state histories in a hidden markov model.
 * 
 * @author bwillard
 *
 * @param <T>
 */
public class HMMTransitionState<T> extends AbstractCloneableSerializable {
  
  private List<WeightedValue<Integer>> stateHistory;
  private HiddenMarkovModel<T> hmm;
  private Integer state;
  private Double stateLogWeight = null;

  public Double getStateLogWeight() {
    return stateLogWeight;
  }

  public void setStateLogWeight(Double stateLogWeight) {
    this.stateLogWeight = stateLogWeight;
  }

  public List<WeightedValue<Integer>> getStateHistory() {
    return stateHistory;
  }

  public HiddenMarkovModel<T> getHmm() {
    return hmm;
  }

  @Override
  public HMMTransitionState<T> clone() {
    HMMTransitionState<T> clone = (HMMTransitionState<T>) super.clone();
    clone.state = new Integer(this.state);
    clone.hmm = this.hmm;
    clone.stateHistory = this.stateHistory;
    clone.stateLogWeight = this.stateLogWeight != null ? new Double(this.stateLogWeight) : null;
    
    return clone;
  }

  @Override
  public String toString() {
    return "HMMTransitionState[state=" + state + ", stateLogWeight="
        + stateLogWeight + "]";
  }

  public HMMTransitionState(HiddenMarkovModel<T> hmm, Integer state) {
    this.hmm = hmm;
    this.state = state;
    this.stateHistory = Lists.newArrayList();
  }

  public HMMTransitionState(HMMTransitionState<T> prevState, Integer newState) {
    this.hmm = prevState.getHmm();
    this.state = newState;
    this.stateHistory = Lists.newArrayList(prevState.stateHistory);
    this.stateHistory.add(DefaultWeightedValue.create(prevState.state, prevState.stateLogWeight));
  }

  public Integer getState() {
    return state;
  }

  public WeightedValue<Integer> getPrevState() {
    return Iterables.getLast(stateHistory);
  }

}
