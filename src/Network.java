import linearalgebra.Matrix;
import linearalgebra.Vector;

import java.util.Arrays;
import java.util.Random;

public class Network {
    Layer[] layers;
    Matrix[] weights; // [layer][Matrix]
    double learning_rate;
    Random random = new Random();
    public Network(int[] architecture, int samples, double learning_rate) {
        //set learning rate
        this.learning_rate = learning_rate;
        // Initialize layers and weights
        // Need to rethink whether or not to include the input layer in this
        //Since we it doesn't have weights and it doesn't get a value because the value is just the input
        layers = new Layer[architecture.length - 1];
        // init our weights array to be one less than the total layers
        // since the input layer doesn't have weights
        //we have three weights matrixes (layer 1, 2, 3)
        //but currently we have four layers (since we count the input layer) with four biases
        weights = new Matrix[architecture.length - 1];
        //a little weird since we start from one, but our layers are still 0 indexed
        //reason being that the input layer is simply the inputs and doesn't have any weights, biases, or computed values
        for(int i = 1; i <= architecture.length - 1; i++) {
            int layerSize = architecture[i];
            layers[i-1] = new Layer(layerSize, samples);
            weights[i-1] = new Matrix(getWeights(layerSize, architecture[i-1]));
            layers[i-1].setBiases(getWeights(layerSize));
        }
    }
    // returns a new array of weights for our matrix
    // not sure if this is the best method, but with our current Matrix package
    // you can't really assign initialValues individually...
    public double[][] getWeights(int row, int column) {
        double[][] weights = new double[row][column];
        for(int r = 0; r < row; r++) {
            for(int c = 0; c < column; c++) {
                weights[r][c] = random.nextGaussian();
            }
        }
        return weights;
    }
    public double[] getWeights(int size) {
        double[] weights = new double[size];
        for(int i = 0; i < size; i++){
            weights[i] = random.nextGaussian();
        }
        return weights;
    }
    //TODO: implement forward pass and sigmoid function - DONE;
    /*
        Forward pass:

        L[l] = W[l] * A[l-1] + B[l];
     */
    public Matrix forwardPass(Matrix input) {
        /*
           First Pass: Weights[1] @ Activation[0](input) + Biases[1]
           Second Pass: Weights[2] @ Activation[1] + Biases[2]
         */
        for(int l = 0; l < this.layers.length - 1; l++){
            if(l == 0) {
                //first pass only use the input...
                this.layers[l].values = weights[l].multiply(input).add(this.layers[l].getBiasesMatrix());
            }
            else {
                this.layers[l].values = new Matrix(sigmoid(weights[l].multiply(this.layers[l-1].values).add(this.layers[l].getBiasesMatrix())));
            }
        }
        //directly perform the calc for the output layer.
        //lets say we have n layers, starting at 0, 1 ... n - 1
        // the above forward pass goes from 0 to < n - 1 (or from 0 to n - 2)
        // the below computes the final layer (layer n - 1)
        this.layers[this.layers.length - 1].values = new Matrix(sigmoid(weights[this.layers.length - 1].multiply(this.layers[this.layers.length - 2].values).add(this.layers[this.layers.length - 1].getBiasesMatrix())));
        return this.layers[this.layers.length - 1].values;
    }
    public double[][] sigmoid(Matrix m){
        double[][] sig = new double[m.getNumRows()][m.getNumColumns()];
        for(int r = 0; r < m.getNumRows(); r++) {
            for(int c = 0; c < m.getNumColumns(); c++) {
                sig[r][c] = 1 / (1 + Math.exp(-1 * m.getEntry(r, c)));
            }
        }
        return sig;
    }

    public double bceLoss(Matrix y_hat, Matrix y_actual) {
        //add check for same shape...
        // add small epsilon value to y_hat to avoid crashes...
        double loss = 0;
        for(int r = 0; r < y_hat.getNumRows(); r++) {
            for(int c = 0; c < y_hat.getNumColumns(); c++) {
                double y = y_actual.getEntry(r, c);
                double yh = y_hat.getEntry(r, c);
                loss += - ((y * Math.log(yh)) + (1 - y)*Math.log(1-yh));
            }
        }
        return ((double) 1 /y_actual.getNumColumns()) * loss;
    }
    /*
        Backpropagation time....
        So, we need to get fancy-upside-down-triangle C by getting the partial derivatives of all the parameters with
        respect to C (loss)
     */
    public void testBackprop(int m, Matrix y, Matrix x) {
        /*
          We want to find fancy upside triangle C.
          Which is [ dC/dW[1], dC/dW[2], dC/dW[3], dC/dB[1], dC/dB[2], dC/dB[3]]

          Lets create our trees....
          so the partial derivative of dC/dW[3] is what?
          well it would be: dL/dC * dC/dW[3].
          A3 = sigmoid(Z3), Z3 = W[3] * A2 + B3
          our network is Loss -> A3 -> Z3 -> A2 -> Z2 -> A1 -> Z1 -> W1 and A0

          so our tree is:
                         C
                         |
                         A3
                         |
                        Z3
                       / | \
                      W3 A2 B3
                         |
                        Z2
                       / | \
                      W2 A1 B2
                         |
                        Z1
                       / | \
                      W1 A0 B1
          If we want to find the partial derivative of W3 with respect to C:
          C -> A3: dC/dA3
          A3 -> Z3: dA3/dZ3
          Z3 -> W3: dZ3/dW3
          so dC/dW3 = dC/dA3 * dA3/dZ3 * dZ3/dW3
          C = bce(A3, y)
          A3 = sigmoid(Z3)
          sigmoid = 1 / 1 + e ^ -x

          Z3 = W3 * A2 + B3
          dZ3/dW3 = A2
          dA3/dZ3 =

          dC/dA^[l] = -1/m (y/A^[l] - (1-y)/(1-A^[l]))
          dA^[l]/dZ^[l] = A^[l](1 - A[l])
          dZ^[l]/dW^[l] = A^[l-1]

          dC/dW^[l] = dC/dA^[l] * dA^[l]/dZ^[l] * dZ^[l]/dW^[l]
          the above can simplify to dC/dW^[l] = dC/dZ[l] * dZ^[l]/dW^[l]

          when we multiply dC/dA^[l] by dA^[l]/dZ^[l] something really cool happens....
          everything simplifies down to 1/m (A^[l] -y)!

          so now we can calculate dC/dW^[l] which is 1/m (A^[l] -y) (A^[l-1])

          but now we have a problem... (A^[l] - y) results in a n^[l] X m matrix
          but A^[l-1] is a n^[l-1] X m matrix...

          we can't do matrix multiplication with these, but luckily they have the shared m term!

          So we can transpose A^[l-1] to make it a m X n^[l-1] matrix!

          now for the code....
          for now we will just do the last layer... which is 3
         */
        //IT WORKS!!!!!!
        //That wasn't quite as bad as I thought lol.... but thats only one layer and only the weights :0
        //will our summation function work? Yes!
        /*
            From the article that I'm following (all code is still written by me,
            with the exception of the linearalegbra library):

            In conclusion, the backprop algorithm is like so:

            Calculate ∂C/∂W^[L] and ∂C/∂b^[L] for the final layer L.
            Calculate the propagator for the penultimate layer L-1 by finding ∂C/∂A^[L-1].
            For all layers l starting from l = L-1, and going until the first layer l=1,
            calculate ∂C/∂W^[l], ∂C/∂b^[l], and the propagator for the next layer ∂C/∂A^[l-1]
            The only caveat for this is that we can substitute A^[0] for X in the code,
            since they are exactly synonymous (they are both the input layer).
         */
        //Matrix prop = W3.transpose().multiply(dCdZ.multiply((double)1/m));
        //now to loop this and get all our values?
        Matrix[] weightUpdates = new Matrix[this.weights.length];
        Matrix[] biasesUpdates = new Matrix[this.weights.length];
        //starting from the final layer...
        Matrix prop = null;
        //This is messy....but it works so...
        //TODO Make this neater. Can probably move some of these operations to functions maybe?
        for(int l = this.layers.length -1; l >= 0; l--) {
            Matrix Al = this.layers[l].values;
            Matrix Al1 = (l) == 0 ? x : this.layers[l - 1].values;
            Matrix Wl = this.weights[l];
            if(l == this.layers.length - 1) {
                Matrix dCdZ = Al.subtract(y).multiply((double) 1 / m);
                Matrix dCdW = dCdZ.multiply(Al1.transpose());
                Matrix dCdB = this.sum(dCdZ);
                weightUpdates[(weightUpdates.length -1) - l] = dCdW;
                biasesUpdates[(biasesUpdates.length -1) - l] = dCdB;
                prop = Wl.transpose().multiply(dCdZ);
            } else {
                Matrix oneminusAl = this.elementWiseSubtractDouble(1, Al);
                Matrix dAdZ = this.elementWiseMultiply(Al, oneminusAl);
                assert prop != null;
                Matrix dCdZ = this.elementWiseMultiply(prop, dAdZ);
                Matrix dCdW = dCdZ.multiply(Al1.transpose());
                Matrix dCdB = this.sum(dCdW);
                weightUpdates[(weightUpdates.length -1) - l] = dCdW;
                biasesUpdates[(biasesUpdates.length -1) - l] = dCdB;
                prop = Wl.transpose().multiply(dCdZ);
            }
        }
        //So now we need to update our weights by taking the values generated above and multiplying everything
//        for(Matrix weight: weightUpdates) {
//            this.printShape(weight);
//        }
        this.updateWeightsBiases(weightUpdates, biasesUpdates);
    }
    public void printShape(Matrix m){
        System.out.println("Shape: " + m.getNumRows() + ", " + m.getNumColumns());
    }
    public void updateWeightsBiases(Matrix[] weightUpdates, Matrix[] biasesUpdates) {
        //loop through our layers and update the weight and biases...
        for(int l = this.weights.length - 1; l >= 0; l--) {
            //first make our new matrix...
            Matrix lrWeight = weightUpdates[(weightUpdates.length - 1) - l].multiply(this.learning_rate);
            Matrix weightUpdate = this.elementWiseSubtract(this.weights[l], lrWeight);
            this.weights[l] = weightUpdate;
            Matrix lrBias = biasesUpdates[(weightUpdates.length - 1) - l].multiply(this.learning_rate);
            Matrix biasUpdate = this.elementWiseSubtract(this.layers[l].getBiasesMatrixUpdate(), lrBias);
            this.layers[l].setBiases(biasUpdate);
        }
    }
    //sum up a given matrix by its rows to end up with an n^l X 1 matrix
    public Matrix sum(Matrix m) {
        double[][] summation = new double[m.getNumRows()][1];
        for(int r = 0; r < m.getNumRows(); r++) {
            double rowSum = 0;
            for(int c = 0; c < m.getNumColumns(); c++) {
                rowSum += m.getEntry(r, c);
            }
            summation[r][0] = rowSum;
        }
        return new Matrix(summation);
    }
    public Matrix elementWiseSubtractDouble(double subtract, Matrix m) {
        double[][] oneMinusA2 = new double[m.getNumRows()][m.getNumColumns()];
        for(int r = 0; r < m.getNumRows(); r++) {
            for(int c = 0; c < m.getNumColumns(); c++) {
                oneMinusA2[r][c] = subtract - m.getEntry(r, c);
            }
        }
        return new Matrix(oneMinusA2);
    }
    public Matrix elementWiseSubtract(Matrix m, Matrix m2) {
        assert (m.getNumRows() == m2.getNumRows()) && (m.getNumColumns() == m2.getNumColumns());
        double[][] result = new double[m.getNumRows()][m.getNumColumns()];
        for(int r = 0; r < m.getNumRows(); r++) {
            for(int c = 0; c < m.getNumColumns(); c++) {
                result[r][c] = m.getEntry(r, c) - m2.getEntry(r, c);
            }
        }
        return new Matrix(result);
    }
    public Matrix elementWiseMultiply(Matrix m, Matrix m2) {
        assert (m.getNumRows() == m2.getNumRows()) && (m.getNumColumns() == m2.getNumColumns());
        double[][] result = new double[m.getNumRows()][m.getNumColumns()];
        for(int r = 0; r < m.getNumRows(); r++) {
            for(int c = 0; c < m.getNumColumns(); c++) {
                result[r][c] = m.getEntry(r, c) * m2.getEntry(r, c);
            }
        }
        return new Matrix(result);
    }
    public Matrix fill(double  d, int r, int c) {
        double[][] fill = new double[r][c];
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++){
                fill[i][j] = d;
            }
        }
        return new Matrix(fill);
    }
}
