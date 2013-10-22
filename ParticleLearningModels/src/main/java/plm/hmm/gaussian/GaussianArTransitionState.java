package plm.hmm.gaussian;

import java.util.List;

import plm.hmm.HmmTransitionState;
import plm.hmm.StandardHMM;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.statslibextensions.util.ObservedValue;
import com.statslibextensions.util.CountedWeightedValue;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.CloneableSerializable;
import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.WeightedValue;

/**
 * A simple class for tracking hidden state histories in a hidden markov model
 * while tracking a gaussian state
 * 
 * @author bwillard
 *
 * @param <T>
 */
public class GaussianArTransitionState extends HmmTransitionState<Double, StandardHMM<Double>> {
  
  private UnivariateGaussian suffStat;
  private UnivariateGaussian priorPredSuffStats; 

  public UnivariateGaussian getSuffStat() {
    return suffStat;
  }

  public void setSuffStat(UnivariateGaussian suffStat) {
    this.suffStat = suffStat;
  }

  @Override
  public GaussianArTransitionState clone() {
    GaussianArTransitionState clone = (GaussianArTransitionState) super.clone();
    clone.suffStat = this.suffStat;
    return clone;
  }

  @Override
  public String toString() {
    return "HMMTransitionState[t=" + this.getTime() + ","
      + (this.prevState != null ? this.prevState.getClassId() : "NA") + " -> " 
      + this.classId 
      + ", suffStat=" + this.suffStat
      + ", (" + this.stateLogWeight + ")]";
  }

  public GaussianArTransitionState(StandardHMM<Double> hmm, Integer state, ObservedValue<Double,Void> obs, UnivariateGaussian suffStat) {
    super(hmm, state, obs);
    this.suffStat = suffStat;
  }

  public GaussianArTransitionState(GaussianArTransitionState prevState, 
      StandardHMM<Double> hmm,
      Integer newState, ObservedValue<Double,Void> obs, UnivariateGaussian suffStat) {
    super(prevState, hmm, newState, obs);
    this.suffStat = suffStat;
  }

  public UnivariateGaussian getPriorPredSuffStats() {
    return this.priorPredSuffStats;
  }

  public void setPriorPredSuffStats(UnivariateGaussian univariateGaussian) {
    this.priorPredSuffStats = univariateGaussian;
  }

}
