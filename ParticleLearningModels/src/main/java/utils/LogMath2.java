package utils;

import gov.sandia.cognition.math.LogMath;

public class LogMath2 extends LogMath {

  /**
   * Adds two log-domain values. It uses a trick to prevent numerical overflow and underflow.
   * <br>See <a href='http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf'>paper</a> 
   * @param logX The first log-domain value (log(x)). Must be the same basis as logY.
   * @param logY The second log-domain value (log(y)). Must be the same basis as logX.
   * @return The log of x plus y (log(x + y)).
   */
  public static double add(final double logX, final double logY) {
    final double maxVal;
    final double minVal;
    if (logX > logY) {
      maxVal = logX;
      minVal = logY;
    } else if (logY > logX) {
      maxVal = logY;
      minVal = logX;
    } else {
      // Since x == y, we have log(x + y) = log(x * 2) = log(x) + log(2).
      return logX + LOG_2;
    }
    final double z = minVal - maxVal;
    if (z <= 18d) 
      return maxVal + Math.log1p(Math.exp(z));
    else if (z <= 33.3d)
      return maxVal + z + Math.exp(-z);
    else
      return maxVal + z;
  }

  /**
   * Subtracts two log-domain values. It uses a trick to prevent numerical overflow and underflow.
   * <br>See <a href='http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf'>paper</a> 
   * 
   * @param logX The first log-domain value (log(x)). Must be the same basis as logY.
   * @param logY The second log-domain value (log(y)). Must be the same basis as logX.
   * @return The log of x minus y (log(x - y)).
   */
  public static double subtract(final double logX, final double logY) {
    if (logX > logY) {
      final double z = logY - logX;
      if (0d < z && z <= LOG_2)
        return logX + Math.log(-Math.expm1(z));
      else
        return logX + Math.log1p(-Math.exp(z));
    } else if (logY > logX) {
      // Since y > x, we will have a log of a negative number, which
      // does not exist.
      return Double.NaN;
    } else if (logX == Double.POSITIVE_INFINITY) {
      // Infinity minus infinity is normally a NaN.
      return Double.NaN;
    } else {
      // Since x == y, we have log(x - y) = log(0), which is negative
      // infinity.
      return LOG_0;
    }
  }

}
