package plm.hmm.gaussian;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.ObjectUtil;

import java.util.List;

import plm.hmm.DlmHiddenMarkovModel;

import com.statslibextensions.util.ObservedValue;

/**
 * Particle encapsulating a sampled transition in an HMM.  This particular one
 * tracks sufficient statistics for an AR(1) DLM, i.e.
 * \[
 *  y_t = F_t x_t + v_t, \, v_t \sim N(0, V_t/\phi)  \\
 *  x_t = \alpha + \beta x_{t-1} + w_t, \, w_t \sim N(0, W_t/\phi) 
 * \] 
 * In this class we define \(\psi = [\alpha, \beta] \sim N(m^\psi, C^\psi)\), and track its
 * sufficient stats., and \(\phi \sim IG(n, S)\)'s (the scale), \(x_t \sim N(m_t^x, C_t^x)\)'s 
 * (the state) sufficient stats., as well.
 * 
 * TODO: This is really a particle object; should make it a subclass thereof.
 * 
 * @author Brandon Willard 
 *
 */
public class GaussianArHpTransitionState extends DlmTransitionState {
  
  private static final long serialVersionUID = 3244374890924323039L;

  protected InverseGammaDistribution scaleSS;
  protected List<MultivariateGaussian> psiSS;
  protected Vector stateSample;
  protected double scaleSample;
  
  public GaussianArHpTransitionState(
      GaussianArHpTransitionState prevState,
      DlmHiddenMarkovModel hmm, Integer classId,
      ObservedValue<Vector,Void> data, MultivariateGaussian state, Vector stateSample, 
      InverseGammaDistribution scaleSS, List<MultivariateGaussian> psiSS,
      double scaleSample) {
    super(prevState, hmm, classId, data, state);
    this.scaleSS = scaleSS;
    this.psiSS = psiSS;
    this.stateSample = stateSample;
    this.scaleSample = scaleSample;
  }

  public GaussianArHpTransitionState(DlmHiddenMarkovModel hmm, Integer classId,
      ObservedValue<Vector,Void> data, MultivariateGaussian state, Vector stateSample,
      InverseGammaDistribution scaleSS, List<MultivariateGaussian> psiSS,
      double scaleSample) {
    super(hmm, classId, data, state);
    this.stateSample = stateSample;
    this.scaleSS = scaleSS;
    this.psiSS = psiSS;
    this.scaleSample = scaleSample;
  }

  @Override
  public GaussianArHpTransitionState clone() {
    GaussianArHpTransitionState clone = (GaussianArHpTransitionState) super.clone();
    clone.stateSample = this.stateSample.clone();
    clone.scaleSample = this.scaleSample;
    clone.scaleSS = this.scaleSS.clone();
    clone.psiSS = ObjectUtil.cloneSmartElementsAsArrayList(this.psiSS);
    return clone;
  }

  public Vector getStateSample() {
    return stateSample;
  }

  public InverseGammaDistribution getScaleSS() {
    return scaleSS;
  }

  public List<MultivariateGaussian> getPsiSS() {
    return psiSS;
  }

  public double getScaleSample() {
    return scaleSample;
  }

}
