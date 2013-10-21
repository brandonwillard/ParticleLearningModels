package plm.hmm;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.CloneableSerializable;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

import java.util.List;
import java.util.Random;

import com.statslibextensions.util.ObservedValue;

public interface GenericHMM<ResponseType, StateType, StateDist extends ComputableDistribution<StateType>> extends CloneableSerializable {

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

  /**
   * Returns p(s_t | y^t) for all y_t in observations.
   * 
   * @return
   */
//  public List<WeightedValue<Vector>> getForwardProbabilities(List<ResponseType> observations);

  public List<SimHmmObservedValue<ResponseType, StateType>> sample(Random random, int T);

  /**
   * Returns p(s_t | y^T) for all y_t in observations.
   * 
   * @return
   */
//  public List<WeightedValue<Vector>> getBackwardProbabilities(List<Vector> stateLikelihoods,
//      List<WeightedValue<Vector>> forwardProbs);

  public int getNumStates();

  public Matrix getTransitionProbability();

  // FIXME this interface really doesn't makes sense...
  public StateDist getEmissionFunction(StateDist state, int classId);
  
  public Vector getClassMarginalProbabilities();

}
