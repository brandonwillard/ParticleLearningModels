package plm.hmm.gaussian;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.ObjectUtil;

import java.util.List;

import plm.hmm.DlmHiddenMarkovModel;

import com.statslibextensions.util.ObservedValue;

public class GaussianArHpTransitionState extends DlmTransitionState {
  
  protected InverseGammaDistribution scaleSS;
  protected List<MultivariateGaussian> systemOffsetsSS;
  protected Vector stateSample;
  protected double scaleSample;
  
  public GaussianArHpTransitionState(
      GaussianArHpTransitionState prevState,
      DlmHiddenMarkovModel hmm, Integer classId,
      ObservedValue<Vector,Void> data, MultivariateGaussian state, Vector stateSample, 
      InverseGammaDistribution scaleSS, List<MultivariateGaussian> systemSS,
      double scaleSample) {
    super(prevState, hmm, classId, data, state);
    this.scaleSS = scaleSS;
    this.systemOffsetsSS = systemSS;
    this.stateSample = stateSample;
    this.scaleSample = scaleSample;
  }

  public GaussianArHpTransitionState(DlmHiddenMarkovModel hmm, Integer classId,
      ObservedValue<Vector,Void> data, MultivariateGaussian state, Vector stateSample,
      InverseGammaDistribution scaleSS, List<MultivariateGaussian> systemSS,
      double scaleSample) {
    super(hmm, classId, data, state);
    this.stateSample = stateSample;
    this.scaleSS = scaleSS;
    this.systemOffsetsSS = systemSS;
    this.scaleSample = scaleSample;
  }

  @Override
  public GaussianArHpTransitionState clone() {
    GaussianArHpTransitionState clone = (GaussianArHpTransitionState) super.clone();
    clone.stateSample = this.stateSample.clone();
    clone.scaleSample = this.scaleSample;
    clone.scaleSS = this.scaleSS.clone();
    clone.systemOffsetsSS = ObjectUtil.cloneSmartElementsAsArrayList(this.systemOffsetsSS);
    return clone;
  }

  public Vector getStateSample() {
    return stateSample;
  }

  public InverseGammaDistribution getScaleSS() {
    return scaleSS;
  }

  public List<MultivariateGaussian> getSystemOffsetsSS() {
    return systemOffsetsSS;
  }

  public double getScaleSample() {
    return scaleSample;
  }

}
