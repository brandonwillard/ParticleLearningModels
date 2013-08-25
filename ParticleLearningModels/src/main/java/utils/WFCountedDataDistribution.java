package utils;

import gov.sandia.cognition.statistics.DataDistribution;

import java.util.Map;

/**
 * Just a wrapper that carries water-filling debug information.
 * 
 * @author bwillard
 *
 * @param <T>
 */
public class WFCountedDataDistribution<T> extends
    CountedDataDistribution {
  
  boolean wasWaterFillingApplied = false;

  public WFCountedDataDistribution(boolean isLogScale) {
    super(isLogScale);
    // TODO Auto-generated constructor stub
  }

  public WFCountedDataDistribution(DataDistribution other,
    boolean isLogScale) {
    super(other, isLogScale);
    // TODO Auto-generated constructor stub
  }

  public WFCountedDataDistribution(int initialCapacity,
    boolean isLogScale) {
    super(initialCapacity, isLogScale);
    // TODO Auto-generated constructor stub
  }

  public WFCountedDataDistribution(Iterable data, boolean isLogScale) {
    super(data, isLogScale);
    // TODO Auto-generated constructor stub
  }

  public WFCountedDataDistribution(Map map, double total,
    boolean isLogScale) {
    super(map, total, isLogScale);
    // TODO Auto-generated constructor stub
  }

  public boolean wasWaterFillingApplied() {
    return wasWaterFillingApplied;
  }

  public void setWasWaterFillingApplied(boolean wasWaterFillingApplied) {
    this.wasWaterFillingApplied = wasWaterFillingApplied;
  }

}
