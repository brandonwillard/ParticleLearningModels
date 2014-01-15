package plm.util.logit.fruehwirth;

import gov.sandia.cognition.math.matrix.Vector;

public class LogitTrueState {
  protected Vector state;
  protected double upperUtility;
  protected double lowerUtility;
  protected double ev1UpperSample;

  public LogitTrueState(Vector state, double upperUtility, 
      double ev1UpperSample, double lowerUtility) {
    this.state = state;
    this.upperUtility = upperUtility;
    this.ev1UpperSample = ev1UpperSample;
    this.lowerUtility = lowerUtility;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("LogitTrueState [state=").append(this.state)
        .append(", upperUtility=").append(this.upperUtility)
        .append(", ev1UpperSample=").append(this.ev1UpperSample)
        .append(", lowerUtility=").append(this.lowerUtility)
        .append("]");
    return builder.toString();
  }
  
}