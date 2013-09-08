package hmm;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.DeterministicDistribution;
import gov.sandia.cognition.util.WeightedValue;

import java.util.List;
import java.util.Random;

/**
 * HMM without state dependencies.
 * @author bwillar0
 *
 * @param <ResponseType>
 */
public class StandardHMM<ResponseType> implements BasicHMM<ResponseType, Integer, DataDistribution<Integer>> {
  
  protected HiddenMarkovModel<ResponseType> hmm;

  public StandardHMM(Vector initialProbs, Matrix transProbs, List<? extends ComputableDistribution<ResponseType>> emissions) {
    hmm = new HiddenMarkovModel<>(initialProbs, transProbs, emissions);
  }

  @Override
  public List<WeightedValue<DataDistribution<Integer>>> getForwardProbabilities(
      List<ResponseType> observations) {
    return null;
  }

  @Override
  public List<SimHmmObservedValue<ResponseType, Integer>> sample(
      Random random, int T) {
    return null;
  }

  @Override
  public List<WeightedValue<DataDistribution<Integer>>> getBackwardProbabilities(
      List<ResponseType> observations) {
    return null;
  }


}
