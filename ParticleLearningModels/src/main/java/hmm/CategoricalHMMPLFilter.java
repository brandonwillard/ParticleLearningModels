package hmm;

import gov.sandia.cognition.collection.IntegerSpan;
import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.ComputableDistribution;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DiscreteSamplingUtil;
import gov.sandia.cognition.statistics.bayesian.AbstractParticleFilter;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.MultivariateStudentTDistribution;
import gov.sandia.cognition.statistics.distribution.NormalInverseWishartDistribution;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.DefaultWeightedValue.WeightComparator;
import gov.sandia.cognition.util.ObjectUtil;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.log4j.Logger;
import org.paukov.combinatorics.Factory;
import org.paukov.combinatorics.Generator;
import org.paukov.combinatorics.ICombinatoricsVector;

import utils.CountedDataDistribution;
import utils.LogMath2;
import utils.MutableDoubleCount;
import utils.SamplingUtils;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.math.DoubleMath;
import com.google.common.primitives.Doubles;

/**
 * A Particle Learning filter for a multivariate Gaussian Dirichlet Process.
 * 
 * @author bwillard
 * 
 */
public class CategoricalHMMPLFilter extends AbstractParticleFilter<Double, HMMTransitionState<Double>> {

  final Logger log = Logger
      .getLogger(CategoricalHMMPLFilter.class);

  public class CategoricalHMMPLUpdater extends AbstractCloneableSerializable
      implements
        Updater<Double, HMMTransitionState<Double>> {

    final private HiddenMarkovModel<Double> hmm; 
    final private Random rng;

    public CategoricalHMMPLUpdater(HiddenMarkovModel<Double> hmm, Random rng) {
      this.hmm = hmm;
      this.rng = rng;
    }

    @Override
    public double computeLogLikelihood(HMMTransitionState<Double> particle, Double observation) {
      return Double.NaN;
    }
    

    /**
     * TODO
     * Constructs an initial particle set by spreading the
     * prior density uniquely (when possible).
     */
    public DataDistribution<HMMTransitionState<Double>> spreadDistribution(int numParticles) {

      PriorityQueue<WeightedValue<Integer>> initialStates = new PriorityQueue<WeightedValue<Integer>>(
          this.hmm.getNumStates(),
          new Comparator<WeightedValue<Integer>> () {
            @Override
            public int compare(WeightedValue<Integer> o1,
              WeightedValue<Integer> o2) {
              return Double.compare(o1.getWeight(), o2.getWeight());
            }
          });

      for (int i = 0; i < this.hmm.getNumStates(); i++) {
        final double stateProb = this.hmm.getInitialProbability().getElement(i);
        initialStates.add(DefaultWeightedValue.create(i, stateProb));
      }


      final CountedDataDistribution<HMMTransitionState<Double>> initialParticles =
          new CountedDataDistribution<>(numParticles, true);
          
      return null;

    }

    @Override
    public DataDistribution<HMMTransitionState<Double>> createInitialParticles(int numParticles) {
      final CountedDataDistribution<HMMTransitionState<Double>> initialParticles =
          new CountedDataDistribution<>(numParticles, true);
      for (int i = 0; i < numParticles; i++) {
        final int sampledState = DiscreteSamplingUtil.sampleIndexFromProbabilities(
            this.rng, this.hmm.getInitialProbability());
        final HMMTransitionState<Double> particle =
            new HMMTransitionState<Double>(this.hmm, sampledState); 

        final double logWeight = -Math.log(numParticles); 
        particle.setStateLogWeight(logWeight); 
        initialParticles.increment(particle, logWeight);
      }
      return initialParticles;
    }

    @Override
    public HMMTransitionState<Double> update(HMMTransitionState<Double> previousParameter) {
      return previousParameter;
    }

  }

  /**
   * Expands the forward probabilities until numParticles
   * paths are reached, then returns the resulting weighed sample path.
   * 
   * @param hmm
   * @param observations
   * @param numParticles
   * @return
   */
  public static <T> TreeSet<HMMTransitionState<T>> expandForwardProbabilities(
      final HiddenMarkovModel<T> hmm, List<T> observations) {

    ExposedHMM<T> eHmm = new ExposedHMM<T>(hmm);

    /*
     * Compute Baum-Welch smoothed distribution
     */
    ArrayList<Vector> obsLikelihoodSequence = eHmm.computeObservationLikelihoods(observations);
    final ArrayList<WeightedValue<Vector>> forwardProbabilities = eHmm.computeForwardProbabilities(obsLikelihoodSequence, true);
    
    List<Vector> jointProbs = Lists.newArrayList();
    for (int i = 0; i < forwardProbabilities.size()-1; i++) {
      WeightedValue<Vector> input = forwardProbabilities.get(i);
      final Vector prod = hmm.getTransitionProbability().times(input.getValue());
      jointProbs.add(prod.scale(1d/prod.norm1()));
    }
    jointProbs.add(Iterables.getLast(forwardProbabilities).getValue());

    Integer[] states = new Integer[hmm.getNumStates()];
    for (int i = 0; i < hmm.getNumStates(); i++) {
      states[i] = i;
    }
    ICombinatoricsVector<Integer> initialVector = Factory.createVector(
        states);
    Generator<Integer> gen = Factory.createPermutationWithRepetitionGenerator(initialVector, 
        observations.size());

    /*
     * Iterate through possible state permutations and find their
     * log likelihoods via the Baum-Welch results above.
     */
    TreeSet<HMMTransitionState<T>> orderedDistribution = Sets.newTreeSet(
        new Comparator<HMMTransitionState<T>>() {
          @Override
          public int compare(HMMTransitionState<T> o1,
            HMMTransitionState<T> o2) {
            final int compVal = Double.compare(o1.getStateLogWeight(), o2.getStateLogWeight());
            return compVal == 0 ? 1 : compVal;
          }
        });
    for (ICombinatoricsVector<Integer> combination : gen) {

      HMMTransitionState<T> currentState = null;
      double logWeightOfState = 0d;
      for (int i = 0; i < combination.getSize(); i++) {
        Vector smoothedProbsAtTime = Iterables.get(jointProbs, i);
        final int stateAtTime = combination.getVector().get(i);
        // TODO assuming it's normalized, is that true?
        final double logWeightAtTime = Math.log(smoothedProbsAtTime.getElement(stateAtTime));

        logWeightOfState += logWeightAtTime;
        if (currentState == null) {
          currentState = new HMMTransitionState<T>(hmm, stateAtTime);
        } else {
          currentState = new HMMTransitionState<T>(currentState, stateAtTime);
        }
        currentState.setStateLogWeight(logWeightOfState);
      }
      orderedDistribution.add(currentState);
    }

    return orderedDistribution;
  }

  public CategoricalHMMPLFilter(HiddenMarkovModel<Double> hmm, Random rng) {
    this.setUpdater(new CategoricalHMMPLUpdater(hmm, rng));
    this.setRandom(rng);
  }

  @Override
  public void update(DataDistribution<HMMTransitionState<Double>> target, Double data) {

    /*
     * Compute prior predictive log likelihoods for resampling.
     */
    double particleTotalLogLikelihood = Double.NEGATIVE_INFINITY;
    final List<Double> logLikelihoods = Lists.newArrayList();
    final List<HMMTransitionState<Double>> particleSupport = Lists.newArrayList();
    for (final HMMTransitionState<Double> particle : target.getDomain()) {
      final HiddenMarkovModel<Double> hmm = particle.getHmm();
      
      final int particleCount = ((CountedDataDistribution)target).getCount(particle);
      int i = 0;

      final double particlePriorLogLik = target.getLogFraction(particle);
      for(ComputableDistribution<Double> f : particle.getHmm().getEmissionFunctions()) {
        final double transStateLogLik = f.getProbabilityFunction().logEvaluate(data)
            + particlePriorLogLik 
            + Math.log(hmm.getTransitionProbability().getElement(i, particle.getState()));

        logLikelihoods.addAll(Collections.nCopies(particleCount, transStateLogLik));
//        particleSupport.addAll(Collections.nCopies(particleCount, new HMMTransitionState<Double>(particle, i)));
        /*
         * Just to be safe...
         */
        for (int k = 0; k < particleCount; k++) { 
          particleSupport.add(new HMMTransitionState<Double>(particle, i));
        }

        particleTotalLogLikelihood = LogMath2.add(particleTotalLogLikelihood, transStateLogLik + 
            Math.log(particleCount));
        i++;
      }
    }

    /*
     * Water-filling resample, for a smoothed predictive set
     */
    final CountedDataDistribution<HMMTransitionState<Double>> resampledParticles =
        SamplingUtils.waterFillingResample(Doubles.toArray(logLikelihoods), particleTotalLogLikelihood, 
            particleSupport, this.random, this.numParticles);

    /*
     * Propagate
     */
    final CountedDataDistribution<HMMTransitionState<Double>> updatedDist = new CountedDataDistribution<>(true);
    for (final Entry<HMMTransitionState<Double>, MutableDouble> entry: resampledParticles.asMap().entrySet()) {
      final HMMTransitionState<Double> updatedEntry = entry.getKey().clone();
      updatedEntry.setStateLogWeight(entry.getValue().doubleValue());
      updatedDist.set(updatedEntry, entry.getValue().doubleValue(), ((MutableDoubleCount)entry.getValue()).count);
    }

    Preconditions.checkState(updatedDist.getTotalCount() == this.numParticles);
    target.clear();
    target.incrementAll(updatedDist);
    Preconditions.checkState(((CountedDataDistribution)target).getTotalCount() == this.numParticles);
  }

}
