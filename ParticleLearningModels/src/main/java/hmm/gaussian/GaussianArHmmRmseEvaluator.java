package hmm.gaussian;

import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.DataDistribution;
import hmm.HmmTransitionState;
import hmm.BasicHMM.SimHmmObservedValue;
import hmm.HmmTransitionState.ResampleType;
import au.com.bytecode.opencsv.CSVWriter;

public class GaussianArHmmRmseEvaluator {

  public static final String evaluatorType = "rmse";
  protected RingAccumulator<MutableDouble> runningRate = 
      new RingAccumulator<MutableDouble>();
  protected final String modelId;
  protected final CSVWriter writer;
  
  public GaussianArHmmRmseEvaluator(String modelId, CSVWriter writer) {
    this.modelId = modelId;
    this.writer = writer;
  }

  public <T extends HmmTransitionState<Double>> void evaluate(
      int replication, SimHmmObservedValue<Vector, Vector> obs,
      DataDistribution<T> distribution) {

    final double trueState = obs.getState().getElement(0);
    RingAccumulator<MutableDouble> stateMean = new RingAccumulator<MutableDouble>();
    for (T particle : distribution.getDomain()) {
      GaussianArTransitionState gParticle = (GaussianArTransitionState) particle;
      final double particleWeight = distribution.getFraction(particle);
      stateMean.accumulate(new MutableDouble(gParticle.getSuffStat().getMean() 
          * particleWeight));
    }

    final double rmse = Math.abs(stateMean.getSum().value - trueState);
    runningRate.accumulate(new MutableDouble(rmse));

    String[] line = {
        Integer.toString(replication), 
        Long.toString(obs.getTime()), 
        this.modelId,
        evaluatorType, 
        distribution.getMaxValueKey().getResampleType().toString(), 
        Double.toString(rmse)};
    writer.writeNext(line);
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