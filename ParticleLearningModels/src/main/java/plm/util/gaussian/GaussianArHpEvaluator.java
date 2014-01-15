package plm.util.gaussian;

import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.List;

import plm.gaussian.GaussianArHpWfParticle;
import plm.hmm.GenericHMM;
import plm.hmm.GenericHMM.SimHmmObservedValue;
import plm.hmm.gaussian.GaussianArHpTransitionState;
import plm.hmm.HmmTransitionState;
import au.com.bytecode.opencsv.CSVWriter;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.statslibextensions.util.ObservedValue;
import com.statslibextensions.util.ObservedValue.SimObservedValue;

public class GaussianArHpEvaluator {

  public static final String evaluatorType = "psi and sigma2 learning rmse";
  protected RingAccumulator<MutableDouble> runningStateRmse = 
      new RingAccumulator<MutableDouble>();
  protected RingAccumulator<MutableDouble> runningPsiRmse = 
      new RingAccumulator<MutableDouble>();
  protected RingAccumulator<MutableDouble> runningSigma2Rmse = 
      new RingAccumulator<MutableDouble>();

  protected RingAccumulator<MutableDouble> runningStateErrRate = 
      new RingAccumulator<MutableDouble>();
  protected RingAccumulator<MutableDouble> runningPsiErrRate = 
      new RingAccumulator<MutableDouble>();
  protected RingAccumulator<MutableDouble> runningSigma2ErrRate = 
      new RingAccumulator<MutableDouble>();

  protected final String modelId;
  protected final CSVWriter writer;
  protected final Vector truePsi;
  protected final double trueSigma2;
  private double stateLastRmse;
  private double psiLastRmse;
  private double sigma2LastRmse;
  private Vector stateLastMean;
  private Vector psiLastMean;
  private double sigma2LastMean;
  private double stateLastErrRate;
  private double psiLastErrRate;
  private double sigma2LastErrRate;
  
  public GaussianArHpEvaluator(String modelId, Vector truePsi, double trueSigma2, 
    CSVWriter writer) {
    this.modelId = modelId;
    this.writer = writer;
    this.truePsi = truePsi;
    this.trueSigma2 = trueSigma2;
  }

  public <N, T extends GaussianArHpWfParticle> void evaluate(
      int replication, SimObservedValue<Vector, ?, Vector> obs, DataDistribution<T> distribution) {

    RingAccumulator<Vector> stateAvg = new RingAccumulator<Vector>();
    RingAccumulator<Vector> psiAvg = new RingAccumulator<Vector>();
    RingAccumulator<MutableDouble> sigma2Avg = new RingAccumulator<MutableDouble>();
    for (T particle : distribution.getDomain()) {
      final double particleWeight = distribution.getFraction(particle);
      // stupid hack for a java bug
      final Object pobj = particle;
      Preconditions.checkState(pobj instanceof GaussianArHpWfParticle);
      GaussianArHpWfParticle gParticle = (GaussianArHpWfParticle) pobj;

      MultivariateGaussian state = gParticle.getState();
      stateAvg.accumulate(state.getMean().scale(particleWeight));

      MultivariateGaussian psi = gParticle.getPsiSS();
      psiAvg.accumulate(psi.getMean().scale(particleWeight));
      
      double sigma2Mean = gParticle.getSigma2SS().getMean();
      sigma2Avg.accumulate(new MutableDouble(sigma2Mean * particleWeight));

    }
    
    this.stateLastMean = stateAvg.getSum();
    this.psiLastMean = psiAvg.getSum();
    this.sigma2LastMean = sigma2Avg.getSum().doubleValue();

    this.stateLastRmse = stateLastMean.minus(obs.getTrueState()).norm2();
    this.psiLastRmse = psiLastMean.minus(this.truePsi).norm2();
    this.sigma2LastRmse = Math.abs(this.trueSigma2 - sigma2LastMean);
    runningStateRmse.accumulate(new MutableDouble(stateLastRmse));
    runningPsiRmse.accumulate(new MutableDouble(psiLastRmse));
    runningSigma2Rmse.accumulate(new MutableDouble(sigma2LastRmse));

    this.stateLastErrRate = stateLastRmse/obs.getTrueState().norm2(); 
    this.psiLastErrRate = psiLastRmse/this.truePsi.norm2(); 
    this.sigma2LastErrRate = sigma2LastRmse/this.trueSigma2; 
    runningStateErrRate.accumulate(new MutableDouble(stateLastErrRate));
    runningPsiErrRate.accumulate(new MutableDouble(psiLastErrRate));
    runningSigma2ErrRate.accumulate(new MutableDouble(sigma2LastErrRate));

    if (writer != null) {
      String[] line = {
          Integer.toString(replication), 
          Long.toString(obs.getTime()), 
          this.modelId,
          "psi", 
          distribution.getMaxValueKey().getResampleType().toString(), 
          Double.toString(psiLastRmse)};
      writer.writeNext(line);

      String[] line2 = {
          Integer.toString(replication), 
          Long.toString(obs.getTime()), 
          this.modelId,
          "sigma2", 
          distribution.getMaxValueKey().getResampleType().toString(), 
          Double.toString(sigma2LastRmse)};
      writer.writeNext(line2);

      String[] line3 = {
          Integer.toString(replication), 
          Long.toString(obs.getTime()), 
          this.modelId,
          "state", 
          distribution.getMaxValueKey().getResampleType().toString(), 
          Double.toString(stateLastRmse)};
      writer.writeNext(line3);
    }
    
  }

  public double getStateLastErrRate() {
    return stateLastErrRate;
  }

  public double getPsiLastErrRate() {
    return psiLastErrRate;
  }

  public double getSigma2LastErrRate() {
    return sigma2LastErrRate;
  }

  public Vector getStateLastMean() {
    return stateLastMean;
  }

  public Vector getPsiLastMean() {
    return psiLastMean;
  }

  public double getSigma2LastMean() {
    return sigma2LastMean;
  }

  public double getRunningStateErrRate() {
    return runningStateErrRate.getMean().doubleValue();
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("GaussianArHpEvaluator [modelId=");
    builder.append(getModelId());
    builder.append(",\n truePsi=");
    builder.append(getTruePsi());
    builder.append(",\n trueSigma2=");
    builder.append(getTrueSigma2());
    builder.append(",\n stateLastMean=");
    builder.append(getStateLastMean());
    builder.append(",\n psiLastMean=");
    builder.append(getPsiLastMean());
    builder.append(",\n sigma2LastMean=");
    builder.append(getSigma2LastMean());
    builder.append(",\n stateLastRmse=");
    builder.append(getStateLastRmse());
    builder.append(",\n psiLastRmse=");
    builder.append(getPsiLastRmse());
    builder.append(",\n sigma2LastRmse=");
    builder.append(getSigma2LastRmse());
    builder.append(",\n runningStateErrRate=");
    builder.append(getRunningStateErrRate());
    builder.append(",\n runningPsiErrRate=");
    builder.append(getRunningPsiErrRate());
    builder.append(",\n runningSigma2ErrRate=");
    builder.append(getRunningSigma2ErrRate());
    builder.append(",\n runningStateRmse=");
    builder.append(getRunningStateRmse());
    builder.append(",\n runningSigma2Rmse=");
    builder.append(getRunningSigma2Rmse());
    builder.append(",\n runningPsiRmse=");
    builder.append(getRunningPsiRmse());
    builder.append("]");
    return builder.toString();
  }

  public double getRunningPsiErrRate() {
    return runningPsiErrRate.getMean().doubleValue();
  }

  public double getRunningSigma2ErrRate() {
    return runningSigma2ErrRate.getMean().doubleValue();
  }

  public static String getEvaluatortype() {
    return evaluatorType;
  }

  public Vector getTruePsi() {
    return truePsi;
  }

  public double getTrueSigma2() {
    return trueSigma2;
  }

  public double getStateLastRmse() {
    return stateLastRmse;
  }

  public double getPsiLastRmse() {
    return psiLastRmse;
  }

  public double getSigma2LastRmse() {
    return sigma2LastRmse;
  }

  public double getRunningStateRmse() {
    return this.runningStateRmse.getMean().doubleValue();
  }

  public double getRunningSigma2Rmse() {
    return this.runningSigma2Rmse.getMean().doubleValue();
  }

  public double getRunningPsiRmse() {
    return this.runningPsiRmse.getMean().doubleValue();
  }

  public String getModelId() {
    return modelId;
  }

  public CSVWriter getWriter() {
    return writer;
  }

}