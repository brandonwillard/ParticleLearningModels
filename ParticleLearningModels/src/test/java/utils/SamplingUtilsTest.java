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

    final double logAlpha = SamplingUtils.findLogAlpha(testLogWeights, 3);

    assertEquals(Math.log(5d), logAlpha, 1e-7);
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

}
