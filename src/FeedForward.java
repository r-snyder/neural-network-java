import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public class FeedForward extends Layer{
    Random rand;
    String name;
    public FeedForward(int inputSize, int outputSize, ActivationFunction activate, String name) {
        this.name = name;
        this.activation = activate;
        rand = new Random();
        //use He initialization...
        double bound = Math.sqrt((2.0 / inputSize));
        this.weights = SimpleMatrix.random_DDRM(outputSize, inputSize, -bound, bound, this.rand);
        //biases are a matrix with shape outputSize, 1...
        this.biases = SimpleMatrix.filled(outputSize, 1, 0);
    }
    @Override
    /*
        Basic Feed Forward calculation
        Z[L] = W[L] * A[L-1] + B[L]
        A[L] = activation(Z[L])
     */
    SimpleMatrix forward(SimpleMatrix input) {
        SimpleMatrix biasesBroadcast = new SimpleMatrix(this.biases);
        biasesBroadcast.reshape(this.biases.getNumRows(), input.getNumCols());
        this.cachedOutput = activation.apply(this.weights.mult(input).plus(biasesBroadcast));
        this.cachedInput = input;

        return this.cachedOutput;
    }

    SimpleMatrix forwardNoGrad(SimpleMatrix input) {
        SimpleMatrix biasesBroadcast = new SimpleMatrix(this.biases);
        biasesBroadcast.reshape(this.biases.getNumRows(), input.getNumCols());

        return activation.apply(this.weights.mult(input).plus(biasesBroadcast));
    }

    @Override
    /*
        Backword Propagation calculation
        Takes in a gradient which will usually be the previous layers calculation
        for feed forward its dAdZ (activation derivative w.r.t Z)
        to dCdZ.
     */
    SimpleMatrix backward(SimpleMatrix gradient) {
        SimpleMatrix dAdZ = this.activation.derivative(this.cachedOutput);
        SimpleMatrix dCdZ = dAdZ.elementMult(gradient);
        this.gradientWeight = dCdZ.mult(this.cachedInput.transpose());
        this.gradientBiases = this.sum(dCdZ);

        return this.weights.transpose().mult(dCdZ);
    }

    SimpleMatrix backwardOutput(SimpleMatrix gradient) {
        this.gradientWeight = gradient.mult(this.cachedInput.transpose());
        this.gradientBiases = this.sum(gradient);

        return this.weights.transpose().mult(gradient);
    }
}
