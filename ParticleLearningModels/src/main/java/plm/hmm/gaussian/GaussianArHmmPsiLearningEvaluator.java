package plm.hmm.gaussian;

import java.util.Arrays;
import java.util.List;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.statslibextensions.util.ObservedValue;

import plm.hmm.GenericHMM;
import plm.hmm.HmmTransitionState;
import plm.hmm.GenericHMM.SimHmmObservedValue;
import plm.hmm.HmmTransitionState.ResampleType;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import au.com.bytecode.opencsv.CSVWriter;

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
      Preconditions.checkState(particle instanceof GaussianArHpTransitionState);
      GaussianArHpTransitionState gParticle = (GaussianArHpTransitionState) particle;
      List<MultivariateGaussian> psis = gParticle.getSystemOffsetsSS();

      // FIXME TODO how to order/identify psis?  sort by magnitude of offset?
      int i = 0;
      for (MultivariateGaussian psi : psis) {
        stateMeans.get(i).accumulate(psi.getMean().scale(particleWeight));
        i++;
      }
    }

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