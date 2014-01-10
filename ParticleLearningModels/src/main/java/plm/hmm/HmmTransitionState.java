package plm.hmm;

import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.WeightedValue;

import java.util.List;

import com.google.common.collect.Lists;
import com.statslibextensions.util.CountedWeightedValue;
import com.statslibextensions.util.ObservedValue;

/**
 * A simple class for tracking hidden state histories in a hidden markov model.
 * 
 * @author bwillard
 *
 * @param <ResponseType>
 */
public class HmmTransitionState<ResponseType, HmmType extends GenericHMM<ResponseType, ?, ?>> 
  extends AbstractCloneableSerializable {
  
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

  protected List<WeightedValue<Integer>> classHistory;
  protected HmmType hmm;
  protected Integer classId;
  protected Double stateLogWeight = null;
  protected ObservedValue<ResponseType, Void> obs;
  protected HmmTransitionState<ResponseType, HmmType> prevState = null;
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
    return classHistory;
  }

  public HmmType getHmm() {
    return hmm;
  }

  @Override
  public HmmTransitionState<ResponseType, HmmType> clone() {
    HmmTransitionState<ResponseType, HmmType> clone = (HmmTransitionState<ResponseType, HmmType>) super.clone();
    clone.classId = new Integer(this.classId);
    clone.hmm = (HmmType) this.hmm.clone();
    clone.resampleType = this.resampleType;
    clone.prevState = this.prevState;
    clone.classHistory = this.classHistory;
    clone.stateLogWeight = this.stateLogWeight != null ? new Double(this.stateLogWeight) : null;
    clone.obs = this.obs;
    return clone;
  }

  @Override
  public String toString() {
    return "HMMTransitionState[t=" + this.obs.getTime() + ","
      + (this.prevState != null ? this.prevState.getClassId() : "NA") + " -> " 
      + this.classId 
      + ", (" + this.stateLogWeight + ")]";
  }

  public HmmTransitionState(HmmType hmm, 
      Integer state, ObservedValue<ResponseType, Void> data) {
    this.obs = data;
    this.hmm = hmm;
    this.classId = state;
    this.classHistory = Lists.newArrayList();
  }

  public HmmTransitionState(HmmTransitionState<ResponseType, HmmType> prevState, 
      HmmType hmm, Integer newState, 
      ObservedValue<ResponseType, Void> data) {
    this.obs = data;
    this.hmm = hmm;
    this.classId = newState;
    /*
     * Clone the previous state so that we can safely remove
     * the reference to its predecessor.
     */
    this.prevState = prevState.clone();
    this.prevState.prevState = null;

    this.classHistory = Lists.newArrayList(prevState.classHistory);
    this.classHistory.add(CountedWeightedValue.create(prevState.classId, prevState.stateLogWeight));
  }

  public Integer getClassId() {
    return classId;
  }

  public HmmTransitionState<ResponseType, HmmType> getPrevState() {
    return this.prevState;
  }

  public void
      setResampleType(ResampleType rType) {
    this.resampleType = rType;
  }

  public ResampleType getResampleType() {
    return this.resampleType;
  }

  public ObservedValue<ResponseType, Void> getObservation() {
    return this.obs;
  }

}
