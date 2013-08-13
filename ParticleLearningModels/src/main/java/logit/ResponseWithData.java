package logit;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;

public class ResponseWithData {

  final Vector response;
  final Matrix data;

  public ResponseWithData(Vector response, Matrix data) {
    super();
    this.response = response;
    this.data = data;
  }

  public Matrix getData() {
    return data;
  }

  public Vector getResponse() {
    return response;
  }

  @Override
  public String toString() {
    return "ResponseWithData [response=" + response + ", data=" + data + "]";
  }

}
