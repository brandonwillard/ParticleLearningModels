package hmm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;

import utils.SamplingUtils;
import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.UnivariateStatisticsUtil;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrix;
import gov.sandia.cognition.math.matrix.mtj.decomposition.CholeskyDecompositionMTJ;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.ObjectUtil;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

public class DlmHiddenMarkovModel extends AbstractCloneableSerializable implements BasicHMM<Vector, Vector, MultivariateGaussian> {

  protected List<KalmanFilter> stateFilters;
  protected Vector marginalClassProbs;
  protected Matrix classTransProbs;
  
  public DlmHiddenMarkovModel(List<KalmanFilter> stateFilters, Vector marginalClassProbs, Matrix classTransProbs) {
    this.stateFilters = stateFilters;
    this.marginalClassProbs = marginalClassProbs;
    this.classTransProbs = classTransProbs;
  }

  public List<KalmanFilter> getStateFilters() {
    return stateFilters;
  }

  public void setStateFilters(List<KalmanFilter> stateFilters) {
    this.stateFilters = stateFilters;
  }
  
  /**
   * Returns p(s_t | y^t) for all y_t in observations.
   * 
   * @return
   */
  @Override
  public List<WeightedValue<MultivariateGaussian>> getForwardProbabilities(List<Vector> observations) {
    return null;
  }

  /**
   * Returns p(s_t | y^T) for all y_t in observations.
   * 
   * @return
   */
  @Override
  public List<WeightedValue<MultivariateGaussian>> getBackwardProbabilities(List<Vector> observations) {
    return null;
  }

  @Override
  public DlmHiddenMarkovModel clone() {
    DlmHiddenMarkovModel clone = (DlmHiddenMarkovModel) super.clone();
    clone.classTransProbs = this.classTransProbs.clone();
    clone.marginalClassProbs = this.marginalClassProbs.clone();
    clone.stateFilters = ObjectUtil.cloneSmartElementsAsArrayList(this.stateFilters);
    return clone;
  }
  

  /**
   * Sample a trajectory up to time T.
   * 
   * @param random
   * @param numSamples
   * @return
   */
  @Override
  public List<SimHmmObservedValue<Vector, Vector>> sample(Random random, int T) {
    List<SimHmmObservedValue<Vector, Vector>> results = Lists.newArrayList();

    int currentClass = DiscreteSamplingUtil.sampleIndexFromProbabilities(random, this.marginalClassProbs);
    Vector currentState = null;
    for (int i = 0; i < T; i++) {

      Vector classProbs = (currentState == null) ? this.marginalClassProbs :
        this.classTransProbs.getColumn(currentClass);

      currentClass = DiscreteSamplingUtil.sampleIndexFromProbabilities(random, classProbs);

      KalmanFilter currentFilter = this.stateFilters.get(currentClass);
      if (currentState == null) {
        currentState = currentFilter.createInitialLearnedObject().sample(random);
      } else {
        final Matrix G = currentFilter.getModel().getA();
        Matrix modelCovSqrt = CholeskyDecompositionMTJ.create(
            (DenseMatrix) currentFilter.getModelCovariance()).getR();
        currentState = MultivariateGaussian.sample(G.times(currentState), modelCovSqrt, random);
      }

      final Matrix F = currentFilter.getModel().getC();
      Vector observationMean = F.times(currentState);
      Matrix measurementCovSqrt = CholeskyDecompositionMTJ.create(
          (DenseMatrix) currentFilter.getMeasurementCovariance()).getR();
      Vector observation = MultivariateGaussian.sample(observationMean, 
          measurementCovSqrt, random);
  
      results.add(new SimHmmObservedValue<Vector, Vector>(i, currentClass, 
          currentState.clone(), observation));
    }
    
    return results;
  }


}
