package plm.hmm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;
import com.statslibextensions.statistics.ExtSamplingUtils;

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

public class DlmHiddenMarkovModel extends AbstractCloneableSerializable implements GenericHMM<Vector, Vector, MultivariateGaussian> {

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
   * Note: this method will add the filter's current input to the
   * initial (and all other) states, so if you set the model input, the
   * offset/input will be added twice to the initial state. 
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
      currentState.plusEquals(currentFilter.getModel().getB().times(
          currentFilter.getCurrentInput()));

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

  @Override
  public int getNumStates() {
    return this.marginalClassProbs.getDimensionality();
  }

  @Override
  public MultivariateGaussian getEmissionFunction(MultivariateGaussian state, int classId) {
    KalmanFilter kf = this.stateFilters.get(classId);
    final Vector mean = kf.getModel().getC().times(state.getMean());
    final Matrix cov = kf.getModel().getC().times(state.getCovariance())
        .times(kf.getModel().getC().transpose())
        .plus(kf.getMeasurementCovariance());
    final MultivariateGaussian likelihood = new MultivariateGaussian(
        mean, cov); 
    return likelihood;
  }

  @Override
  public Matrix getTransitionProbability() {
    return this.classTransProbs;
  }

  @Override
  public Vector getClassMarginalProbabilities() {
    return this.marginalClassProbs;
  }


}
