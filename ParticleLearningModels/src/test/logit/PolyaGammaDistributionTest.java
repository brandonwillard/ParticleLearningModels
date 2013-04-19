package logit;

import static org.junit.Assert.*;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;

import java.util.Random;

import junit.framework.Assert;

import org.junit.Test;

public class PolyaGammaDistributionTest {

  /**
   * Compare a large sample mean to the analytical mean of PG(1,z), over
   * a symmetric interval of z. (Note that the analytical mean cannot be evaluated at zero)
   * <br>
   * The accuracy observed from the implementation we ported (BayesLogit in R) is around
   * 1e-3 to 1e-6.  Our results are expected to be within the same range.
   */
  @Test
  public void test() {
    
    Random rng = new Random(2502035l);
    final int numOfSamples = 100000;
    final double intervalHalf = 100d;
    final int domainSteps = 100;
    double z = -intervalHalf;
    for (int j = 0; j < domainSteps; j++) {
      z += (2d * intervalHalf)/domainSteps;
      /*
       * Can't evaluate the mean at zero using this formula.
       */
      if (z == 0d)
        continue;
      System.out.println("z=" + z);
      UnivariateGaussian.SufficientStatistic averager = new UnivariateGaussian.SufficientStatistic();
      for (int i = 0; i < numOfSamples; i++) {
        final double sample = PolyaGammaDistribution.sample(z, rng);
        averager.update(sample);
      }
      final double pg1Mean = 1d/(2d*z) * Math.tanh(z/2d);
      System.out.println("mean estimation error=" + (pg1Mean - averager.getMean()));
      assertEquals(pg1Mean, averager.getMean(), 1e-3);
    }
    System.out.println("done");
  }

}
