package utils;

import static org.junit.Assert.*;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.PoissonDistribution;
import gov.sandia.cognition.util.Pair;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.junit.Test;

import com.google.common.collect.DiscreteDomains;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Range;
import com.google.common.collect.Ranges;
import com.google.common.primitives.Doubles;

public class SamplingUtilsTest {

  @Test
  public void testFindLogAlpha1() {
    double[] testLogWeights =
        new double[] { Math.log(5d / 11d), Math.log(3d / 11d),
            Math.log(2d / 11d), Math.log(1d / 11d) };

    final double logAlpha1 =
        SamplingUtils.findLogAlpha(testLogWeights, 0d, 1);

    double pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal =
          LogMath2.add(pTotal,
              Math.min(testLogWeights[i] + logAlpha1, 0d));
    }
    assertEquals(Math.log(1), pTotal, 1e-7);
    assertEquals(0, logAlpha1, 1e-7);

    final double logAlpha2 =
        SamplingUtils.findLogAlpha(testLogWeights, 0d, 2);

    pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal =
          LogMath2.add(pTotal,
              Math.min(testLogWeights[i] + logAlpha2, 0d));
    }
    assertEquals(Math.log(2), pTotal, 1e-7);
    assertEquals(0.6931471805599d, logAlpha2, 1e-7);

    final double logAlpha3 =
        SamplingUtils.findLogAlpha(testLogWeights, 0d, 3);

    pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal =
          LogMath2.add(pTotal,
              Math.min(testLogWeights[i] + logAlpha3, 0d));
    }
    assertEquals(Math.log(3), pTotal, 1e-7);
    assertEquals(1.29928298413d, logAlpha3, 1e-7);
  }
  
  @Test
  public void testFindLogAlpha2() {
    double[] testLogWeights =
        new double[] { 
        -4.758895070648572, -2.561670493312353, 
        -4.954271557120856, -2.757046979784638, 
        -2.757046979784638, -4.954271557120856, 
        -4.954271557120856, -2.757046979784638, 
        -2.757046979784638, -4.954271557120856 };

    SamplingUtils.logNormalize(testLogWeights);

    final double logAlpha1 =
        SamplingUtils.findLogAlpha(testLogWeights, 0d, 5);

    double pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal =
          LogMath2.add(pTotal,
              Math.min(testLogWeights[i] + logAlpha1, 0d));
    }
    assertEquals(Math.log(5), pTotal, 1e-7);
  }

  /**
   * 10 weights all about 1/10, then 10 are about zero.  Since
   * we can't really water-fill at this high a precision, we should
   * allow it to resample, which would give the same results (given
   * that there are ten weights that are significantly greater than
   * the other ten, we'll end up taking those regardless).  
   */
  @Test
  public void testFindLogAlpha3() {
//    double[] testLogWeights2 = { 
//        -2.3025850929940397, -2.3025850929940397, 
//        -2.3025850929940397, -2.3025850929940397, -2.3025850929940397, 
//        -2.3025850929940397, -2.3025850929940397, -2.3025850929940397, 
//        -2.3025850929940397, -2.3025850929940397, -341.22111709254136, 
//        -341.22111709254136, -341.22111709254136, -341.22111709254136, 
//        -341.22111709254136, -341.22111709254136, -341.22111709254136, 
//        -341.22111709254136, -341.22111709254136, -341.22111709254136 }; 
//
//    double s1 = 0d;
//    for (int i = 0; i < testLogWeights2.length; i++) {
//      s1 = LogMath2.subtract(s1, testLogWeights2[i]);
//    }

    double[] testLogWeights =
        new double[] { 
        -341.22111709254136, -2.3025850929940397, -341.22111709254136, 
         -2.3025850929940397, -341.22111709254136, -2.3025850929940397, 
         -341.22111709254136, -2.3025850929940397, -341.22111709254136, 
         -2.3025850929940397, -341.22111709254136, -2.3025850929940397, 
         -341.22111709254136, -2.3025850929940397, -341.22111709254136, 
         -2.3025850929940397, -341.22111709254136, -2.3025850929940397, 
         -341.22111709254136, -2.3025850929940397 };

    final double totalLogWeight = 5.995204332975845E-15;

//    SamplingUtils.logNormalize(testLogWeights);

    final double logAlpha1 =
        SamplingUtils.findLogAlpha(testLogWeights, totalLogWeight, 10);

    assertTrue(Double.isInfinite(logAlpha1));
  }

  /**
   * Test basic resample with flat weights
   */
  @Test
  public void testWaterFillingResample1() {
    double[] testLogWeights =
        new double[] { Math.log(1d / 4d), Math.log(1d / 4d),
            Math.log(1d / 4d), Math.log(1d / 4d) };
    String[] testObjects = new String[] { "o1", "o2", "o3", "o4" };

    final Random rng = new Random(123569869l);
    final int N = 2;
    DataDistribution<String> wfResampleResults =
        SamplingUtils.waterFillingResample(testLogWeights, 0d,
            Lists.newArrayList(testObjects), rng, N);

    for (Entry<String, ? extends Number> logWeight : wfResampleResults
        .asMap().entrySet()) {
      assertEquals(-Math.log(N), logWeight.getValue().doubleValue(),
          1e-7);
    }
  }

  /**
   * Test error when < N non-zero weights
   */
  @Test(expected = IllegalStateException.class)
  public void testWaterFillingResample2() {
    double[] testLogWeights =
        new double[] { Double.NEGATIVE_INFINITY,
            Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, 0d };
    String[] testObjects = new String[] { "o1", "o2", "o3", "o4" };

    final Random rng = new Random(123569869l);
    final int N = 2;
    DataDistribution<String> wfResampleResults =
        SamplingUtils.waterFillingResample(testLogWeights, 0d,
            Lists.newArrayList(testObjects), rng, N);

    assertEquals(0d, wfResampleResults.getMaxValue(), 1e-7);
    assertEquals("o4", wfResampleResults.getMaxValueKey());
    final int count =
        ((MutableDoubleCount) wfResampleResults.asMap().get(
            wfResampleResults.getMaxValueKey())).count;
    assertEquals(2, count);
  }

  /**
   * Test basic resample with N non-zero weights TODO: really just checking for
   * flat weights now, need to check more?
   */
  @Test
  public void testWaterFillingResample3() {
    double[] testLogWeights =
        new double[] { Double.NEGATIVE_INFINITY,
            Double.NEGATIVE_INFINITY, Math.log(1d / 2d),
            Math.log(1d / 2d) };
    String[] testObjects = new String[] { "o1", "o2", "o3", "o4" };

    final Random rng = new Random(123569869l);
    final int N = 2;
    DataDistribution<String> wfResampleResults =
        SamplingUtils.waterFillingResample(testLogWeights, 0d,
            Lists.newArrayList(testObjects), rng, N);

    for (Entry<String, ? extends Number> logWeight : wfResampleResults
        .asMap().entrySet()) {
      assertEquals(-Math.log(N), logWeight.getValue().doubleValue(),
          1e-7);
    }
  }

  /**
   * Test water-filling accepts one and resamples the others
   */
  @Test
  public void testWaterFillingResample4() {
    double[] testLogWeights =
        new double[] { Math.log(5d / 11d), Math.log(3d / 11d),
            Math.log(2d / 11d), Math.log(1d / 11d) };
    String[] testObjects = new String[] { "o1", "o2", "o3", "o4" };

    final Random rng = new Random(123569869l);
    final int N = 3;
    DataDistribution<String> wfResampleResults =
        SamplingUtils.waterFillingResample(testLogWeights, 0d,
            Lists.newArrayList(testObjects), rng, N);

    assertEquals(Math.log(5d / 11d), wfResampleResults.getMaxValue(),
        1e-7);
    assertEquals("o1", wfResampleResults.getMaxValueKey());

    final double logAlpha =
        SamplingUtils.findLogAlpha(testLogWeights, 0d, N);

    List<Double> logWeights = Lists.newArrayList();
    for (Entry<String, ? extends Number> entry : wfResampleResults
        .asMap().entrySet()) {
      final double logWeight = entry.getValue().doubleValue();
      logWeights.add(logWeight);
      if (!entry.getKey().equals(wfResampleResults.getMaxValueKey())) {
        assertEquals(-logAlpha, logWeight, 1e-7);
      }
    }

    assertTrue(SamplingUtils.isLogNormalized(logWeights, 1e-7));
  }

  /**
   * Test water-filling accepts two and resamples the others
   */
  @Test
  public void testWaterFillingResample5() {
    double[] testLogWeights =
        new double[] { Math.log(6d / 17d), Math.log(5d / 17d),
            Math.log(3d / 17d), Math.log(2d / 17d),
            Math.log(1d / 17d) };
    String[] testObjects =
        new String[] { "o0", "o1", "o2", "o3", "o4" };

    final Random rng = new Random(123569869l);
    final int N = 4;
    DataDistribution<String> wfResampleResults =
        SamplingUtils.waterFillingResample(testLogWeights, 0d,
            Lists.newArrayList(testObjects), rng, N);

    assertEquals(Math.log(6d / 17d), wfResampleResults.getMaxValue(),
        1e-7);
    assertEquals("o0", wfResampleResults.getMaxValueKey());
    DataDistribution<String> tmpResults = wfResampleResults.clone();
    tmpResults.decrement(wfResampleResults.getMaxValueKey(),
        tmpResults.getMaxValue());
    assertEquals(Math.log(5d / 17d), tmpResults.getMaxValue(), 1e-7);
    assertEquals("o1", tmpResults.getMaxValueKey());

    final double logAlpha =
        SamplingUtils.findLogAlpha(testLogWeights, 0d, N);

    double pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal =
          LogMath2.add(pTotal,
              Math.min(testLogWeights[i] + logAlpha, 0d));
    }
    assertEquals(Math.log(N), pTotal, 1e-7);
    assertEquals(1.734601055388d, logAlpha, 1e-7);

    List<Double> logWeights = Lists.newArrayList();
    int i = 0;
    for (Entry<String, ? extends Number> entry : wfResampleResults
        .asMap().entrySet()) {
      final double logWeight = entry.getValue().doubleValue();
      logWeights.add(logWeight);
      if (i > 1) {
        assertEquals(-logAlpha, logWeight, 1e-7);
      }
      i++;
    }

    assertTrue(SamplingUtils.isLogNormalized(logWeights, 1e-7));
  }

  /**
   * Sample one from the no-replace sampler and make sure
   * it gives a good sample distribution, just for a sanity check. 
   */
  @Test
  public void testResampleNoReplace1() {
    double[] logWeights = {Math.log(6d/10d), Math.log(2d/10d), Math.log(1d/10d), Math.log(1d/10d)};
    List<String> domain = Lists.newArrayList("e1", "e2", "e3", "e4");
    Random random = new Random();
    final int numSamples = 1;
    final int R = 3000000;

    CountedDataDistribution<String> sampleDist = new CountedDataDistribution<>(false);
    for (int i = 0; i < R; i++) {
      List<String> result = SamplingUtils.sampleNoReplaceMultipleLogScale(
          logWeights, 0d, domain, random, numSamples);
      sampleDist.incrementAll(result);
    }
    
    int i = 0;
    for (String el : domain) {
      System.out.println(el + "=" + sampleDist.getFraction(el) + "(" + Math.exp(logWeights[i]) + ")");
      assertEquals(logWeights[i], sampleDist.getLogFraction(el), 1e-2);
      i++;
    }
    
  }

  /**
   * Just exploring the water-filling behavior
   */
//  @Test
//  public void testWaterFillingBehavior() {
//
//    List<Double> testLogWeights = Lists.newArrayList();
//    List<Integer> testObjects = Lists.newArrayList();
//    Range<Integer> range = Ranges.closed(0, 100);
//    PoissonDistribution pd = new PoissonDistribution(2d);
//    for (int i : range.asSet(DiscreteDomains.integers())) {
//      testLogWeights.add(pd.getProbabilityFunction().logEvaluate(i));
//      testObjects.add(i);
//    }
//
//    final Random rng = new Random(123569869l);
//    final int N = 4;
//    DataDistribution<Integer> wfResampleResults =
//        SamplingUtils.waterFillingResample(
//            Doubles.toArray(testLogWeights), 0d, testObjects, rng, N);
//
//  }
}
