package utils;

public class ObservedValue<T> {
  
  final protected long time;
  final protected T observedValue;

  public ObservedValue(long time, T observedState) {
    this.time = time;
    this.observedValue = observedState;
  }

  public long getTime() {
    return time;
  }

  public T getObservedValue() {
    return observedValue;
  }

  @Override
  public String toString() {
    return "ObservedValue [time=" + time + ", observedValue=" + observedValue
        + "]";
  }
  
  
}
