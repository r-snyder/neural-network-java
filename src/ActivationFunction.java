import org.ejml.simple.SimpleMatrix;

public abstract class ActivationFunction {

    public abstract SimpleMatrix apply(SimpleMatrix x);
    public abstract SimpleMatrix derivative(SimpleMatrix a);
}
