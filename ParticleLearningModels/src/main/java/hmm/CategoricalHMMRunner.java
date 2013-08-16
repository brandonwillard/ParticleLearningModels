package hmm;

import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;

public class CategoricalHMMRunner {

  public static void main(String[] args) {
    
    DefaultDataDistribution<Double> s1Likelihood = new DefaultDataDistribution<Double>();
    s1Likelihood.increment(0d, 0.5);
    s1Likelihood.increment(1d, 0.5);

    DefaultDataDistribution<Double> s2Likelihood = new DefaultDataDistribution<Double>();
    s2Likelihood.increment(0d, 0.5);
    s2Likelihood.increment(1d, 0.5);
    
    HiddenMarkovModel<Double> hmm = new HiddenMarkovModel<Double>(
        VectorFactory.getDefault().copyArray(new double[] {0.5, 0.5}),
        MatrixFactory.getDefault().copyArray(new double[][] {{0.5, 0.5}, {0.5, 0.5}}),
        Lists.newArrayList(s1Likelihood, s2Likelihood)
        );
    
    final Random rng = new Random(123452388l);
    final int N = 50;
    List<Double> sample = hmm.sample(rng, N);
    
    System.out.println(sample);
    
    CategoricalHMMPLFilter filter = new CategoricalHMMPLFilter(hmm, rng);
    DataDistribution<HMMTransitionState> distribution = filter.createInitialLearnedObject();
    
    for (int i = 0; i < N; i++) {
      filter.update(distribution, sample.get(i));
    }

  }

}
