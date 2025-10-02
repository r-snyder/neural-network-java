import org.ejml.simple.SimpleMatrix;

public class Relu extends ActivationFunction{


    public SimpleMatrix apply(SimpleMatrix z) {
        return this.relu(z);
    }

    public SimpleMatrix derivative(SimpleMatrix a) {
        SimpleMatrix result = new SimpleMatrix(a);
        for (int i = 0; i < a.getNumElements(); i++) {
            //changing to LeakyReLU
            result.set(i, a.get(i) > 0 ? 1.0 : 0.02);
        }
        return result;
    }
    protected SimpleMatrix relu(SimpleMatrix z) {
        SimpleMatrix  relu = new SimpleMatrix(z.getNumRows(), z.getNumCols());
        for(int r = 0; r < z.getNumRows(); r++) {
            for(int c = 0; c < z.getNumCols(); c++) {
                relu.set(r, c, Math.max(0.02d, z.get(r, c)));
            }
        }
        return relu;
    }
}
