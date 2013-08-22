package hmm;

public class ObservedState {
  
  private final long time;
  private final int observedState;

  public ObservedState(long time, int observedState) {
    this.time = time;
    this.observedState = observedState;
  }

  public long getTime() {
    return time;
  }

  public int getObservedState() {
    return observedState;
  }
}
