package utils;

import static org.junit.Assert.*;
import gov.sandia.cognition.util.Pair;

import java.util.List;
import java.util.Random;

import org.junit.Test;

import com.google.common.collect.Lists;

public class SamplingUtilsTest {

  @Test
  public void testFindLogAlpha1() {
    double[] testLogWeights = new double[] { Math.log(5d/11d),
                                             Math.log(3d/11d),
                                             Math.log(2d/11d),
                                             Math.log(1d/11d)};
    
    final double logAlpha1 = SamplingUtils.findLogAlpha(testLogWeights, 1);

    double pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal = LogMath2.add(pTotal, Math.min(testLogWeights[i] + logAlpha1, 0d));
    }
    assertEquals(Math.log(1), pTotal, 1e-7);
    assertEquals(0, logAlpha1, 1e-7);

    final double logAlpha2 = SamplingUtils.findLogAlpha(testLogWeights, 2);

    pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal = LogMath2.add(pTotal, Math.min(testLogWeights[i] + logAlpha2, 0d));
    }
    assertEquals(Math.log(2), pTotal, 1e-7);
    assertEquals(0.6931471805599d, logAlpha2, 1e-7);

    final double logAlpha3 = SamplingUtils.findLogAlpha(testLogWeights, 3);
    
    pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal = LogMath2.add(pTotal, Math.min(testLogWeights[i] + logAlpha3, 0d));
    }
    assertEquals(Math.log(3), pTotal, 1e-7);
    assertEquals(1.29928298413d, logAlpha3, 1e-7);
  }

  /**
   * Test basic resample with flat weights
   */
  @Test
  public void testWaterFillingResample1() {
    double[] testLogWeights = new double[] { -Math.log(1d/4d),
                                             -Math.log(1d/4d),
                                             -Math.log(1d/4d),
                                             -Math.log(1d/4d)};
    String[] testObjects = new String[] {"o1", "o2", "o3", "o4"};

    final Random rng = new Random(123569869l);
    final int N = 2;
    Pair<List<Double>, List<String>> wfResampleResults = SamplingUtils.waterFillingResample(
        testLogWeights, 0d, Lists.newArrayList(testObjects), rng, N);

    for (Double logWeight : wfResampleResults.getFirst()) {
      assertEquals(-Math.log(N), logWeight, 1e-7);
    }
  }

  /**
   * Test basic resample with < N non-zero weights
   * TODO: really just checking for flat weights now, need to check more?
   */
  @Test
  public void testWaterFillingResample2() {
    double[] testLogWeights = new double[] { Double.NEGATIVE_INFINITY,
                                             Double.NEGATIVE_INFINITY,
                                             Double.NEGATIVE_INFINITY,
                                             0d};
    String[] testObjects = new String[] {"o1", "o2", "o3", "o4"};

    final Random rng = new Random(123569869l);
    final int N = 2;
    Pair<List<Double>, List<String>> wfResampleResults = SamplingUtils.waterFillingResample(
        testLogWeights, 0d, Lists.newArrayList(testObjects), rng, N);

    for (Double logWeight : wfResampleResults.getFirst()) {
      assertEquals(-Math.log(N), logWeight, 1e-7);
    }
  }

  /**
   * Test basic resample with  N non-zero weights
   * TODO: really just checking for flat weights now, need to check more?
   */
  @Test
  public void testWaterFillingResample3() {
    double[] testLogWeights = new double[] { Double.NEGATIVE_INFINITY,
                                             Double.NEGATIVE_INFINITY,
                                             Math.log(1d/2d),
                                             Math.log(1d/2d)};
    String[] testObjects = new String[] {"o1", "o2", "o3", "o4"};

    final Random rng = new Random(123569869l);
    final int N = 2;
    Pair<List<Double>, List<String>> wfResampleResults = SamplingUtils.waterFillingResample(
        testLogWeights, 0d, Lists.newArrayList(testObjects), rng, N);

    for (Double logWeight : wfResampleResults.getFirst()) {
      assertEquals(-Math.log(N), logWeight, 1e-7);
    }
  }

  /**
   * Test water-filling accepts one and resamples the others
   */
  @Test
  public void testWaterFillingResample4() {
    double[] testLogWeights = new double[] { Math.log(5d/11d),
                                             Math.log(3d/11d),
                                             Math.log(2d/11d),
                                             Math.log(1d/11d)};
    String[] testObjects = new String[] {"o1", "o2", "o3", "o4"};

    final Random rng = new Random(123569869l);
    final int N = 3;
    Pair<List<Double>, List<String>> wfResampleResults = SamplingUtils.waterFillingResample(
        testLogWeights, 0d, Lists.newArrayList(testObjects), rng, N);

    assertEquals(Math.log(5d/11d), wfResampleResults.getFirst().get(0), 1e-7);
    assertEquals("o1", wfResampleResults.getSecond().get(0));

    final double logAlpha = SamplingUtils.findLogAlpha(testLogWeights, N);

    for (int i = 1; i < wfResampleResults.getFirst().size(); i++) {
      final double logWeight = wfResampleResults.getFirst().get(i);
      assertEquals(-logAlpha, logWeight, 1e-7);
    }
    
    assertTrue(SamplingUtils.isLogNormalized(wfResampleResults.getFirst(), 1e-7));
  }

  /**
   * Test water-filling accepts two and resamples the others
   */
  @Test
  public void testWaterFillingResample5() {
    double[] testLogWeights = new double[] { Math.log(6d/17d), Math.log(5d/17d),
                                             Math.log(3d/17d),
                                             Math.log(2d/17d),
                                             Math.log(1d/17d)};
    String[] testObjects = new String[] {"o0", "o1", "o2", "o3", "o4"};

    final Random rng = new Random(123569869l);
    final int N = 4;
    Pair<List<Double>, List<String>> wfResampleResults = SamplingUtils.waterFillingResample(
        testLogWeights, 0d, Lists.newArrayList(testObjects), rng, N);

    assertEquals(Math.log(6d/17d), wfResampleResults.getFirst().get(0), 1e-7);
    assertEquals("o0", wfResampleResults.getSecond().get(0));
    assertEquals(Math.log(5d/17d), wfResampleResults.getFirst().get(1), 1e-7);
    assertEquals("o1", wfResampleResults.getSecond().get(1));

    final double logAlpha = SamplingUtils.findLogAlpha(testLogWeights, N);
    
    double pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal = LogMath2.add(pTotal, Math.min(testLogWeights[i] + logAlpha, 0d));
    }
    assertEquals(Math.log(N), pTotal, 1e-7);
    assertEquals(1.734601055388d, logAlpha, 1e-7);

    for (int i = 2; i < wfResampleResults.getFirst().size(); i++) {
      final double logWeight = wfResampleResults.getFirst().get(i);
      assertEquals(-logAlpha, logWeight, 1e-7);
    }
    
    assertTrue(SamplingUtils.isLogNormalized(wfResampleResults.getFirst(), 1e-7));
  }

}
