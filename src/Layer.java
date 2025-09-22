import linearalgebra.Matrix;
import linearalgebra.Vector;

public class Layer {
    // initialValues are the results of the matrix multiplication from the forward pass
    // but it won't be a 1d array but a 2d array of size n and m where m is the number of samples.
    double[][] initialValues;
    Matrix values;
    double[] biases;
    int samples;
    int size;

    // Constructor
    public Layer(int size, int samples) {
        this.size = size;
        this.samples = samples;
        this.initialValues = new double[size][samples];
        this.biases = new double[size];
    }

    public double[][] getInitialValues() {
        return initialValues;
    }

    public void setInitialValues(double[][] initialValues) {
        this.initialValues = initialValues;
    }

    public double[] getBiases() {
        return biases;
    }

    public void setBiases(double[] biases) {
        this.biases = biases;
    }

    public void setBiases(Matrix biases) {
        //we need to basically undo the broadcast...
        //this is probably dumb but whatever
        double[] bias = new double[this.biases.length];
        Vector row = biases.getRow(0);
        for(int c = 0; c < row.length(); c++) {
            bias[c] = row.get(c);
        }
        this.biases = bias;
    }
    public Matrix getBiasesMatrix() {
        //we need to broadcast this biases from n, 1 to n, m where m is the number of samples...
        // in our case it needs to go from a 3x1 to a 3x10 which we can get by copying our 3x1 10 times.
        double[][] biasesMatrix = new double[this.biases.length][this.samples];
        for(int j = 0; j < this.samples; j++) {
            for(int i = 0; i < this.biases.length; i++) {
                //this should work. outer loop should loop 10 times (0-9) inner loop should loop 3 times (0-2)
                //so for each sample(M) we have one biases "matrix" but they're combined into one...
                biasesMatrix[i][j] = this.biases[i];
            }
        }
        return new Matrix(biasesMatrix);
    }
    public Matrix getBiasesMatrixUpdate() {
        double[][] biasesMatrix = new double[this.biases.length][1];
        for(int i = 0; i < this.biases.length; i++) {
            biasesMatrix[i][0] = this.biases[i];
        }
        return new Matrix(biasesMatrix);
    }
}
