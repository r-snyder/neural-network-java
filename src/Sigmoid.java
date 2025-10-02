import org.ejml.simple.SimpleMatrix;

public class Sigmoid extends ActivationFunction {

    public SimpleMatrix apply(SimpleMatrix z) {
        return this.sigmoid(z);
    }

    public SimpleMatrix derivative(SimpleMatrix a) {
        //return a.elementMult(a.minus(1));
        return a.elementMult(Layer.elementWiseMinus(a, 1));
    }
    protected SimpleMatrix sigmoid(SimpleMatrix z) {
        SimpleMatrix sig = new SimpleMatrix(z.getNumRows(), z.getNumCols());
        for(int r = 0; r < z.getNumRows(); r++) {
            for(int c = 0; c < z.getNumCols(); c++) {
                sig.set(r, c, (1 / (1 + Math.exp(-1 * z.get(r, c)))));
            }
        }
        return sig;
    }
}
