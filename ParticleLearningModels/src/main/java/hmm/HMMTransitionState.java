package hmm;

import java.util.List;

import utils.WrappedWeightedValue;

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

  private HMMTransitionState<T> prevState = null;


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
    return "HMMTransitionState[" 
      + (this.prevState != null ? this.prevState.getState() : "NA") + " -> " 
      + state 
      + ", (" + stateLogWeight + ")]";
  }

  public HMMTransitionState(HiddenMarkovModel<T> hmm, Integer state) {
    this.hmm = hmm;
    this.state = state;
    this.stateHistory = Lists.newArrayList();
  }

  public HMMTransitionState(HMMTransitionState<T> prevState, Integer newState) {
    this.hmm = prevState.getHmm();
    this.state = newState;
    this.prevState = prevState;
    this.stateHistory = Lists.newArrayList(prevState.stateHistory);
    this.stateHistory.add(WrappedWeightedValue.create(prevState.state, prevState.stateLogWeight));
  }

  public Integer getState() {
    return state;
  }

  public HMMTransitionState<T> getPrevState() {
    return this.prevState;
  }

//  @Override
//  public int hashCode() {
//    final int prime = 31;
//    int result = 1;
//    result = prime * result + ((hmm == null) ? 0 : hmm.hashCode());
//    result =
//        prime * result + ((state == null) ? 0 : state.hashCode());
//    // TODO something about the last state
//    return result;
//  }
//
//  @Override
//  public boolean equals(Object obj) {
//    if (this == obj)
//      return true;
//    if (obj == null)
//      return false;
//    if (getClass() != obj.getClass())
//      return false;
//    HMMTransitionState other = (HMMTransitionState) obj;
//    if (hmm == null) {
//      if (other.hmm != null)
//        return false;
//    } else if (!hmm.equals(other.hmm))
//      return false;
//    if (state == null) {
//      if (other.state != null)
//        return false;
//    } else if (!state.equals(other.state))
//      return false;
//
//    // TODO something about equality with the last state
//
//    return true;
//  }

}
