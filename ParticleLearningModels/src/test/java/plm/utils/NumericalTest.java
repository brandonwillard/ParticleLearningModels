package plm.utils;

import java.util.Arrays;

import org.junit.Test;

import com.statslibextensions.math.ExtLogMath;
import com.statslibextensions.util.ExtSamplingUtils;

public class NumericalTest {

  @Test
  public void test() {

    double[] data = new double[] { 
        -341.22111709254136, -2.3025850929940397, -341.22111709254136, 
         -2.3025850929940397, -341.22111709254136, -2.3025850929940397, 
         -341.22111709254136, -2.3025850929940397, -341.22111709254136, 
         -2.3025850929940397, -341.22111709254136, -2.3025850929940397, 
         -341.22111709254136, -2.3025850929940397, -341.22111709254136, 
         -2.3025850929940397, -341.22111709254136, -2.3025850929940397, 
         -341.22111709254136, -2.3025850929940397 };
    Arrays.sort(data);
    /*
     * Plain ol' Kahan summation
     */
    double expSum = Math.exp(data[0]);
    double expC = 0.0f;
    for (int i = 1; i < data.length; i++) {
      double y = Math.exp(data[i]) - expC;
      double t = expSum + y;
      expC = (t - expSum) - y;
      expSum = t;
    }
    System.out.println("expSum=" + Math.log(expSum));

    double naiveSum = ExtSamplingUtils.logSum(data);

    double kahanSum = data[0];
    double logC = Double.NEGATIVE_INFINITY;
    for (int i = 1; i < data.length; i++) {
      double y = ExtLogMath.subtract(data[i], logC);
      double t = ExtLogMath.add(kahanSum, y);
      logC = ExtLogMath.subtract(ExtLogMath.subtract(t, kahanSum), y);
      kahanSum = t;
    }
    
//    double ftSum = data[0];
//    for (int i = 1; i < data.length; i++) {
//      final double a;
//      final double b;
//      if (Math.abs(ftSum) > Math.abs(data[i])) {
//        a = ftSum;
//        b = data[i];
//      } else {
//        b = ftSum;
//        a = data[i];
//      }
//      double x = LogMath2.add(a, b);
//      double bv = LogMath2.subtract(x, a);
//      double err = LogMath2.subtract(b, bv);
//      System.out.println("ftErr=" + err);
//    }
    
    System.out.println("naiveSum=" + naiveSum);
    System.out.println("kahanSum=" + kahanSum);
  }

}
