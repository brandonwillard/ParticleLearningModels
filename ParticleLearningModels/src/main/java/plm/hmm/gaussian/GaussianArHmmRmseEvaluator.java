package plm.hmm.gaussian;

import plm.hmm.GenericHMM;
import plm.hmm.HmmTransitionState;
import plm.hmm.GenericHMM.SimHmmObservedValue;
import plm.hmm.HmmTransitionState.ResampleType;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import au.com.bytecode.opencsv.CSVWriter;

public class GaussianArHmmRmseEvaluator {

  public static final String evaluatorType = "rmse";
  protected RingAccumulator<MutableDouble> runningRate = 
      new RingAccumulator<MutableDouble>();
  protected final String modelId;
  protected final CSVWriter writer;
  
  public GaussianArHmmRmseEvaluator(String modelId) {
    this.modelId = modelId;
    this.writer = null;
  }

  public GaussianArHmmRmseEvaluator(String modelId, CSVWriter writer) {
    this.modelId = modelId;
    this.writer = writer;
  }

  public <N, H extends GenericHMM<N,?,?>, T extends HmmTransitionState<N, H>> void evaluate(
      int replication, SimHmmObservedValue<Vector, Vector> obs,
      DataDistribution<T> distribution) {

    final Vector trueState = obs.getState();
    RingAccumulator<Vector> stateMean = new RingAccumulator<Vector>();
    for (T particle : distribution.getDomain()) {
      final double particleWeight = distribution.getFraction(particle);
      if (particle instanceof GaussianArTransitionState) {
        GaussianArTransitionState gParticle = (GaussianArTransitionState) particle;
        stateMean.accumulate(VectorFactory.getDefault().copyValues(
            gParticle.getSuffStat().getMean() * particleWeight));
      } else if (particle instanceof GaussianArHpTransitionState) {
        GaussianArHpTransitionState gParticle = (GaussianArHpTransitionState) particle;
        stateMean.accumulate(gParticle.getState().getMean().scale(particleWeight));
      }
    }

    final double rmse = stateMean.getSum().minus(trueState).norm2();
    runningRate.accumulate(new MutableDouble(rmse));

    if (writer != null) {
      String[] line = {
          Integer.toString(replication), 
          Long.toString(obs.getTime()), 
          this.modelId,
          evaluatorType, 
          distribution.getMaxValueKey().getResampleType().toString(), 
          Double.toString(rmse)};
      writer.writeNext(line);
    }
  }

  public RingAccumulator<MutableDouble> getTotalRate() {
    return runningRate;
  }

  public void setWfRunningClassRate(
      RingAccumulator<MutableDouble> rate) {
    this.runningRate = rate;
  }

  public String getModelId() {
    return modelId;
  }

  public CSVWriter getWriter() {
    return writer;
  }

}