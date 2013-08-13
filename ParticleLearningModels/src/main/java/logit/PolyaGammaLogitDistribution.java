package logit;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.distribution.ExponentialDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;

import java.util.Random;

import utils.ScaledInverseGammaCovDistribution;
import utils.StatisticsUtils;

/**
 * 
 * A binary logit model with linear gaussian regression log-odds and Lasso shrinkage.
 * 
 * 
 * The static methods are a port of the sampling method for a Polya-Gamma distribution in the
 * BayesLogit R package.
 * 
 * TODO extend Distribution class, add PDF and moments.
 * 
 * @author bwillard
 * 
 */
public class PolyaGammaLogitDistribution {

  static final protected double TRUNC = 0.64d;
  static final protected double TRUNC_RECIP = 1d / TRUNC;
  static final protected ExponentialDistribution expDist = new ExponentialDistribution(1d);

  /**
   * Partial sum coefficient for term n.
   * 
   * @param n
   * @param x
   * @return
   */
  protected static double a(int n, double x) {
    final double K = (n + 0.5d) * Math.PI;
    double y = 0d;
    if (x > TRUNC) {
      y = K * Math.exp(-0.5d * K * K * x);
    } else if (x > 0d) {
      final double expnt =
          -1.5d * (Math.log(0.5d * Math.PI) + Math.log(x)) + Math.log(K) - 2.0d * (n + 0.5d)
              * (n + 0.5d) / x;
      y = Math.exp(expnt);
      // y = pow(0.5 * Math.PI * x, -1.5) * K * exp( -2.0 * (n+0.5)*(n+0.5) / x);
      // ^- unstable for small x?
    }
    return y;
  }

  /**
   * Truncated exponential at 1.
   * 
   * @param Z
   * @return
   */
  protected static double mass_texpon(double Z) {
    final double t = TRUNC;

    final double fz = 0.125d * Math.PI * Math.PI + 0.5d * Z * Z;
    final double b = Math.sqrt(1d / t) * (t * Z - 1d);
    final double a = Math.sqrt(1d / t) * (t * Z + 1d) * -1d;

    final double x0 = Math.log(fz) + fz * t;
    final double xb = x0 - Z + StatisticsUtils.normalCdf(b, 0d, 1d, true);
    final double xa = x0 + Z + StatisticsUtils.normalCdf(a, 0d, 1d, true);

    final double qdivp = 4d / Math.PI * (Math.exp(xb) + Math.exp(xa));

    return 1d / (1d + qdivp);
  }

  /**
   * CDF of an inverse Gaussian with mean Z.
   * 
   * @param x
   * @param Z
   * @return
   */
  protected static double pigauss(double x, double Z) {
    final double b = Math.sqrt(1d / x) * (x * Z - 1d);
    final double a = Math.sqrt(1d / x) * (x * Z + 1d) * -1d;
    final double y =
        UnivariateGaussian.CDF.evaluate(b, 0d, 1d) + Math.exp(2d * Z)
            * UnivariateGaussian.CDF.evaluate(a, 0d, 1d);
    return y;
  }

  /**
   * Draw a sample from a truncated inverse gaussian IG(mu = 1/z, lambda = 1) over (0, t).
   * 
   * @param Z
   * @param rng
   * @return
   */
  protected static double rtigauss(double Z, Random rng) {
    Z = Math.abs(Z);
    final double t = TRUNC;
    double X = t + 1d;
    if (TRUNC_RECIP > Z) { // mu > t
      double alpha = 0d;
      while (rng.nextDouble() > alpha) {
        // X = t + 1.0;
        // while (X > t)
        // X = 1.0 / r.gamma_rate(0.5, 0.5);
        // Slightly faster to use truncated normal.
        double E1 = expDist.sample(rng);
        double E2 = expDist.sample(rng);
        while (E1 * E1 > 2 * E2 / t) {
          E1 = expDist.sample(rng);
          E2 = expDist.sample(rng);
        }
        X = 1d + E1 * t;
        X = t / (X * X);
        alpha = Math.exp(-0.5d * Z * Z * X);
      }
    } else {
      final double mu = 1d / Z;
      while (X > t) {
        double Y = rng.nextGaussian();
        Y *= Y;
        final double half_mu = 0.5d * mu;
        final double mu_Y = mu * Y;
        X = mu + half_mu * mu_Y - half_mu * Math.sqrt(4d * mu_Y + mu_Y * mu_Y);
        if (rng.nextDouble() > mu / (mu + X)) X = mu * mu / X;
      }
    }
    return X;
  }

  /**
   * Draw a sample from PG(1, Z). <br>
   * From PolyaGamma.h in the BayesLogit package, and based on Devroye's method
   * 
   * @param rng
   * @return
   */
  public static double sample(double Z, Random rng) {
    // Change the parameter.
    final double z = Math.abs(Z) * 0.5d;

    // Now sample 0.25 * J^*(1, Z := Z/2).
    final double fz = 0.125d * Math.PI * Math.PI + 0.5d * z * z;
    // ... Problems with large Z? Try using q_over_p.
    // double p = 0.5 * Math.PI * exp(-1.0 * fz * __TRUNC) / fz;
    // double q = 2 * exp(-1.0 * Z) * pigauss(__TRUNC, Z);

    double X = 0.0d;
    double S = 1.0d;
    double Y = 0.0d;
    // int iter = 0; If you want to keep track of iterations.

    while (true) {
      // if (r.unif() < p/(p+q))
      if (rng.nextDouble() < mass_texpon(z))
        X = TRUNC + expDist.sample(rng) / fz;
      else
        X = rtigauss(z, rng);

      S = a(0, X);
      Y = rng.nextDouble() * S;
      int n = 0;
      boolean go = true;

      // Cap the number of iterations?
      while (go) {
        n++;
        if (n % 2 == 1) {
          S = S - a(n, X);
          if (Y <= S) return 0.25d * X;
        } else {
          S = S + a(n, X);
          if (Y > S) go = false;
        }
      }
      // Need Y <= S in event that Y = S, e.g. when X = 0.
    }
  }

  /*
   * The joint distribution of augmented response (based on indep. PG(1, 0))
   */
  private MultivariateGaussian augmentedResponseDistribution;

  /*
   * Regression coefficients
   */
  private MultivariateGaussian priorBeta;

  private ScaledInverseGammaCovDistribution priorBetaCov;

  /*
   * Keep this value around for debugging.
   */
  private Vector priorPredictiveMean;

  public PolyaGammaLogitDistribution(MultivariateGaussian priorBeta,
      ScaledInverseGammaCovDistribution scaledInverseGammaCovDistribution) {
    this.priorBeta = priorBeta;
    this.priorBetaCov = scaledInverseGammaCovDistribution;
  }

  public MultivariateGaussian getAugmentedResponseDistribution() {
    return augmentedResponseDistribution;
  }

  public MultivariateGaussian getPriorBeta() {
    return priorBeta;
  }

  public ScaledInverseGammaCovDistribution getPriorBetaCov() {
    return priorBetaCov;
  }

  public Vector getPriorPredictiveMean() {
    return this.priorPredictiveMean;
  }

  public void setAugmentedResponseDistribution(MultivariateGaussian augmentedResponseDistribution) {
    this.augmentedResponseDistribution = augmentedResponseDistribution;
  }


  public void setPriorBeta(MultivariateGaussian priorBeta) {
    this.priorBeta = priorBeta;
  }

  public void setPriorBetaCov(ScaledInverseGammaCovDistribution priorBetaCov) {
    this.priorBetaCov = priorBetaCov;
  }

  public void setPriorPredictiveMean(Vector phi) {
    this.priorPredictiveMean = phi;
  }

}
