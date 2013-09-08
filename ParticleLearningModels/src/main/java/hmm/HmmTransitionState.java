package hmm;

import java.util.List;

import utils.ObservedValue;
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
 * @param <ResponseType>
 */
public class HmmTransitionState<ResponseType> extends AbstractCloneableSerializable {
  
  public static enum ResampleType {
    NONE ("none"), 
    WATER_FILLING ("water-filling"), 
    NO_REPLACEMENT ("no-replacement"),
    REPLACEMENT ("replacement");

    private final String name;

    private ResampleType(String name) {
      this.name = name;
    }

    public boolean equalsName(String otherName){
      return (otherName == null)? false : name.equals(otherName);
    }

    public String toString(){
       return name;
    }
  }

  protected List<WeightedValue<Integer>> stateHistory;
  protected HiddenMarkovModel<ResponseType> hmm;
  protected Integer state;
  protected Double stateLogWeight = null;
  protected ObservedValue<ResponseType> obs;
  protected HmmTransitionState<ResponseType> prevState = null;
  protected ResampleType resampleType = ResampleType.NONE;

  public long getTime() {
    return obs.getTime();
  }

  public Double getStateLogWeight() {
    return stateLogWeight;
  }

  public void setStateLogWeight(Double stateLogWeight) {
    this.stateLogWeight = stateLogWeight;
  }

  public List<WeightedValue<Integer>> getStateHistory() {
    return stateHistory;
  }

  public HiddenMarkovModel<ResponseType> getHmm() {
    return hmm;
  }

  @Override
  public HmmTransitionState<ResponseType> clone() {
    HmmTransitionState<ResponseType> clone = (HmmTransitionState<ResponseType>) super.clone();
    clone.state = new Integer(this.state);
    clone.hmm = this.hmm;
    clone.resampleType = this.resampleType;
    clone.prevState = this.prevState;
    clone.stateHistory = this.stateHistory;
    clone.stateLogWeight = this.stateLogWeight != null ? new Double(this.stateLogWeight) : null;
    clone.obs = this.obs;
    return clone;
  }

  @Override
  public String toString() {
    return "HMMTransitionState[t=" + this.obs.getTime() + ","
      + (this.prevState != null ? this.prevState.getState() : "NA") + " -> " 
      + this.state 
      + ", (" + this.stateLogWeight + ")]";
  }

  public HmmTransitionState(HiddenMarkovModel<ResponseType> hmm, Integer state, ObservedValue<ResponseType> data) {
    this.obs = data;
    this.hmm = hmm;
    this.state = state;
    this.stateHistory = Lists.newArrayList();
  }

  public HmmTransitionState(HmmTransitionState<ResponseType> prevState, HiddenMarkovModel<ResponseType> hmm, Integer newState, 
      ObservedValue<ResponseType> data) {
    this.obs = data;
    this.hmm = hmm;
    this.state = newState;
    /*
     * Clone the previous state so that we can safely remove
     * the reference to its predecessor.
     */
    this.prevState = prevState.clone();
    this.prevState.prevState = null;

    this.stateHistory = Lists.newArrayList(prevState.stateHistory);
    this.stateHistory.add(WrappedWeightedValue.create(prevState.state, prevState.stateLogWeight));
  }

  public Integer getState() {
    return state;
  }

  public HmmTransitionState<ResponseType> getPrevState() {
    return this.prevState;
  }

  public void
      setResampleType(ResampleType rType) {
    this.resampleType = rType;
  }

  public ResampleType getResampleType() {
    return this.resampleType;
  }

  public ObservedValue<ResponseType> getObservedValue() {
    return this.obs;
  }

}
