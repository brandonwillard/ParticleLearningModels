package plm.util.hmm.gaussian;

import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.List;

import plm.hmm.GenericHMM;
import plm.hmm.GenericHMM.SimHmmObservedValue;
import plm.hmm.gaussian.GaussianArHpTransitionState;
import plm.hmm.HmmTransitionState;
import au.com.bytecode.opencsv.CSVWriter;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.statslibextensions.util.ObservedValue;

public class GaussianArHmmPsiLearningEvaluator {

  public static final String evaluatorType = "psi learning rmse";
//  protected RingAccumulator<MutableDouble> runningRate = 
//      new RingAccumulator<MutableDouble>();
  protected final String modelId;
  protected final CSVWriter writer;
  protected final List<Vector> truePsis;
  private List<Vector> lastValue;
  
  public GaussianArHmmPsiLearningEvaluator(String modelId, List<Vector> truePsi) {
    this.modelId = modelId;
    this.writer = null;
    this.truePsis = truePsi;
  }

  public GaussianArHmmPsiLearningEvaluator(String modelId, List<Vector> truePsi, CSVWriter writer) {
    this.modelId = modelId;
    this.writer = writer;
    this.truePsis = truePsi;
  }

  public <N, H extends GenericHMM<N,?,?>, T extends HmmTransitionState<N, H>> void evaluate(
      int replication, SimHmmObservedValue<Vector, Vector> obs, DataDistribution<T> distribution) {

    List<RingAccumulator<Vector>> stateMeans = Lists.newArrayList();
    for (int i = 0; i < this.truePsis.size(); i++) {
      stateMeans.add(new RingAccumulator<Vector>());
    }
    for (T particle : distribution.getDomain()) {
      final double particleWeight = distribution.getFraction(particle);
      // stupid hack for a java bug
      final Object pobj = particle;
      Preconditions.checkState(pobj instanceof GaussianArHpTransitionState);
      GaussianArHpTransitionState gParticle = (GaussianArHpTransitionState) pobj;
      List<MultivariateGaussian> psis = gParticle.getPsiSS();

      // FIXME TODO how to order/identify psis?  sort by magnitude of offset?
      int i = 0;
      for (MultivariateGaussian psi : psis) {
        stateMeans.get(i).accumulate(psi.getMean().scale(particleWeight));
        i++;
      }
    }

    // TODO FIXME not implemented!
//    for (Vector truePsi : this.truePsi) {
//      final double rmse = stateMean.getSum().minus(truePsi).norm2();
//      runningRate.accumulate(new MutableDouble(rmse));
//    }
//    final double rmse = Double.NaN;//stateMean.getSum().minus(truePsi).norm2();
//    runningRate.accumulate(new MutableDouble(rmse));
    List<Vector> rawMeans = Lists.newArrayList();
    for (RingAccumulator<Vector> mean : stateMeans) {
      rawMeans.add(mean.getSum().clone());
    }

    if (writer != null) {
      String[] line = {
          Integer.toString(replication), 
          Long.toString(obs.getTime()), 
          this.modelId,
          evaluatorType, 
          distribution.getMaxValueKey().getResampleType().toString(), 
          rawMeans.toString()};
      writer.writeNext(line);
    }
    
    this.lastValue = rawMeans;
  }

  public List<Vector> getTotalRate() {
    return this.lastValue;
  }

  public String getModelId() {
    return modelId;
  }

  public CSVWriter getWriter() {
    return writer;
  }

}