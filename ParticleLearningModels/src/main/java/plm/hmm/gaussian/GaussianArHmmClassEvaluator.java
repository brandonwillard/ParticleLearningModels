package plm.hmm.gaussian;

import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.DataDistribution;
import plm.hmm.GenericHMM;
import plm.hmm.GenericHMM.SimHmmObservedValue;
import plm.hmm.HmmTransitionState;
import plm.hmm.HmmTransitionState.ResampleType;
import au.com.bytecode.opencsv.CSVWriter;

public class GaussianArHmmClassEvaluator {

  public static final String evaluatorType = "classification";
  protected RingAccumulator<MutableDouble> runningRate = 
      new RingAccumulator<MutableDouble>();
  protected final String modelId;
  protected final CSVWriter writer;
  
  public GaussianArHmmClassEvaluator(String modelId, CSVWriter writer) {
    this.modelId = modelId;
    this.writer = writer;
  }

  public <N, H extends GenericHMM<N, ?, ?>, T extends HmmTransitionState<N, H>> void evaluate(
      int replication, SimHmmObservedValue<Vector, Vector> obs,
      DataDistribution<T> distribution) {

    final double x = obs.getClassId();
    RingAccumulator<MutableDouble> classificationRate = new RingAccumulator<MutableDouble>();
    for (T state : distribution.getDomain()) {
      final double wfErr = (x == state.getClassId()) ? 
          distribution.getFraction(state) : 0d;
      classificationRate.accumulate(new MutableDouble(wfErr));
    }

    runningRate.accumulate(new MutableDouble(classificationRate.getSum()));

    ResampleType resampleType = distribution.getMaxValueKey().getResampleType();
    String[] wfClassLine = {
        Integer.toString(replication), 
        Long.toString(obs.getTime()), 
        this.modelId,
        evaluatorType, resampleType.toString(), 
        Double.toString(classificationRate.getSum().value)
       };
    this.writer.writeNext(wfClassLine);
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