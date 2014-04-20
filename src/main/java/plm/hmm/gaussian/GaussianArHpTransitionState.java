package plm.hmm.gaussian;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.ObjectUtil;

import java.util.List;

import plm.hmm.DlmHiddenMarkovModel;

import com.statslibextensions.util.ObservedValue;

// TODO: This is really a particle object; should make it a subclass thereof.
/**
 * Particle encapsulating a sampled transition in an HMM.  This particular one
 * tracks sufficient statistics for the AR(1) DLM in {@link plm.hmm.gaussian.GaussianArHpHmmPLFilter}.
 * </br>
 * 
 * @author Brandon Willard 
 *
 */
public class GaussianArHpTransitionState extends DlmTransitionState {
  
  private static final long serialVersionUID = 3244374890924323039L;

  protected InverseGammaDistribution invScaleSS;
  protected List<MultivariateGaussian> psiSS;
  protected Vector stateSample;
  protected double invScaleSample;
  
  public GaussianArHpTransitionState(
      GaussianArHpTransitionState prevState,
      DlmHiddenMarkovModel hmm, Integer classId,
      ObservedValue<Vector,Void> data, MultivariateGaussian state, Vector stateSample, 
      InverseGammaDistribution invScaleSS, List<MultivariateGaussian> psiSS,
      double invScaleSample) {
    super(prevState, hmm, classId, data, state);
    this.invScaleSS = invScaleSS;
    this.psiSS = psiSS;
    this.stateSample = stateSample;
    this.invScaleSample = invScaleSample;
  }

  public GaussianArHpTransitionState(DlmHiddenMarkovModel hmm, Integer classId,
      ObservedValue<Vector,Void> data, MultivariateGaussian state, Vector stateSample,
      InverseGammaDistribution invScaleSS, List<MultivariateGaussian> psiSS,
      double invScaleSample) {
    super(hmm, classId, data, state);
    this.stateSample = stateSample;
    this.invScaleSS = invScaleSS;
    this.psiSS = psiSS;
    this.invScaleSample = invScaleSample;
  }

  @Override
  public GaussianArHpTransitionState clone() {
    GaussianArHpTransitionState clone = (GaussianArHpTransitionState) super.clone();
    clone.stateSample = this.stateSample.clone();
    clone.invScaleSample = this.invScaleSample;
    clone.invScaleSS = this.invScaleSS.clone();
    clone.psiSS = ObjectUtil.cloneSmartElementsAsArrayList(this.psiSS);
    return clone;
  }

  public Vector getStateSample() {
    return stateSample;
  }

  public InverseGammaDistribution getInvScaleSS() {
    return invScaleSS;
  }

  public List<MultivariateGaussian> getPsiSS() {
    return psiSS;
  }

  public double getInvScaleSample() {
    return invScaleSample;
  }

}
