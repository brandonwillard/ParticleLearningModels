package utils;

public class ObservedValue<T> {
  
  private final long time;
  private final T observedState;

  public ObservedValue(long time, T observedState) {
    this.time = time;
    this.observedState = observedState;
  }

  public long getTime() {
    return time;
  }

  public T getObservedState() {
    return observedState;
  }
}
