package hmm;

import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

import java.util.List;
import java.util.Random;

import utils.ObservedValue;

public interface BasicHMM<ResponseType, StateType, StateDist extends ComputableDistribution<StateType>> {

  public static class SimHmmObservedValue<ResponseType, StateType> extends ObservedValue<ResponseType> {
    final protected int classId;
    final protected StateType state;

    public SimHmmObservedValue(long time, int classId, StateType state, ResponseType value) {
      super(time, value);
      this.classId = classId;
      this.state = state;
    }

    public int getClassId() {
      return classId;
    }

    public StateType getState() {
      return state;
    }

    @Override
    public String toString() {
      return "SimHmmObservedState [classId=" + classId + ", state=" + state
          + ", time=" + time + ", observedValue=" + observedValue + "]";
    }
    
  }

  List<WeightedValue<StateDist>> getForwardProbabilities(List<ResponseType> observations);

  List<SimHmmObservedValue<ResponseType, StateType>> sample(Random random, int T);

  List<WeightedValue<StateDist>> getBackwardProbabilities(List<ResponseType> observations);

}
