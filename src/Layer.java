import org.ejml.simple.SimpleMatrix;

public abstract class Layer {
    protected SimpleMatrix weights, biases, cachedInput, cachedOutput, gradientWeight, gradientBiases;
    protected ActivationFunction activation;

    abstract SimpleMatrix forward(SimpleMatrix input);
    abstract SimpleMatrix forwardNoGrad(SimpleMatrix input);
    abstract SimpleMatrix backward(SimpleMatrix gradient);
    abstract SimpleMatrix backwardOutput(SimpleMatrix gradient);

    void updateWeights(double learningRate) {

        this.weights = this.weights.minus(gradientWeight.scale(learningRate));
    }

    void updateBiases(double learningRate) {
        this.biases = this.biases.minus(gradientBiases.scale(learningRate));
    }
    //sum up a given matrix by its rows to end up with an n^l X 1 matrix
    SimpleMatrix sum(SimpleMatrix m) {
        double[][] summation = new double[m.getNumRows()][1];
        for(int r = 0; r < m.getNumRows(); r++) {
            double rowSum = 0;
            for(int c = 0; c < m.getNumCols(); c++) {
                rowSum += m.get(r, c);
            }
            summation[r][0] = rowSum;
        }
        return new SimpleMatrix(summation);
    }

    public static SimpleMatrix elementWiseMinus(SimpleMatrix a, double b) {
        double[][] oneMinusA2 = new double[a.getNumRows()][a.getNumCols()];
        for(int r = 0; r < a.getNumRows(); r++) {
            for(int c = 0; c < a.getNumCols(); c++) {
                oneMinusA2[r][c] = b - a.get(r, c);
            }
        }
        return new SimpleMatrix(oneMinusA2);
    }
}
