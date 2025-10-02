
import org.ejml.simple.SimpleMatrix;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.util.List;
import java.util.Random;
/*
    TODO Add all this to the github project and set up some tests and github actions,etc
   Features I would like to add:
   * learning rate scheduler - to do next
   * different loss functions
   * SGD
   * batch training optimizations
   * data splitting - DONE
   * better validation and training metrics (ROC, precision, recall) - in progress
   * GPU usage for matrix operations (?)
   * layer abstraction - then this
   * Stratified K-folds
   * Make everything neater!
 */
public class Network {
    List<Layer> layers;
    double learning_rate;
    int samples;
    SimpleMatrix x;
    SimpleMatrix y;

    // Ok time to try to use our new FeedForward layer
    public Network(List<Layer> architecture, SimpleMatrix x, SimpleMatrix y, double learning_rate) {
        //set learning rate
        this.learning_rate = learning_rate;
        this.x = x;
        this.y = y;
        this.samples = x.getNumCols();
        this.layers = architecture;
    }
    /*
        Forward pass:

        L[l] = W[l] * A[l-1] + B[l];
     */
    public SimpleMatrix forwardPass(SimpleMatrix input) {
        /*
           First Pass: Weights[1] @ Activation[0](input) + Biases[1]
           Second Pass: Weights[2] @ Activation[1] + Biases[2]

           In this new case, we have an input layer of size 30
           one hidden layer of size 16
           one output layer
         */
        SimpleMatrix current = input;
        for(Layer layer : this.layers) {
            current = layer.forward(current);
        }
        return current;
    }
    public SimpleMatrix forwardPassNoGrad(SimpleMatrix input) {
        SimpleMatrix current = input;
        for(Layer layer: this.layers) {
            current = layer.forwardNoGrad(current);
        }

        return current;
    }

    // TODO add other activation functions

    // TODO add other loss functions
    public double bceLoss(SimpleMatrix y_hat, SimpleMatrix y_actual) {
        assert (y_hat.getNumRows() == y_actual.getNumRows()) && (y_hat.getNumCols() == y_actual.getNumCols());
        // add small epsilon value to y_hat to avoid crashes...
        double loss = 0;
        double epsilon = 1e-16;  // Small value to prevent log(0)
        for(int r = 0; r < y_hat.getNumRows(); r++) {
            for(int c = 0; c < y_hat.getNumCols(); c++) {
                double y = y_actual.get(r, c);
                // we're getting log(0) from our yh output,  will clamp it between epsilon
                // and  1 - epsilon
                double yh = Math.max(epsilon, Math.min(1 - epsilon, y_hat.get(r, c)));
                loss -= ((y * Math.log(yh)) + (1 - y) * Math.log(1 - yh));
            }
        }
        return ((double) 1 /y_actual.getNumCols()) * loss;
    }

    //check the accuracy of predictions
    //will need to change this based on the dataset and whether its a simple
    //binary classification problem or something more complex
    public double accuracy(SimpleMatrix y_hat, SimpleMatrix y_actual) {
            double total = y_actual.getNumCols();  // Number of samples
            double correct = 0;

            // Only loop through columns (samples), not rows
            for(int c = 0; c < y_hat.getNumCols(); c++){
                int predicted = y_hat.get(0, c) > 0.5 ? 1 : 0;  // Row 0 since single output
                if(predicted == y_actual.get(0, c)) correct++;
            }

            return correct / total;
    }

    /*
        Backpropagation time....
        So, we need to get fancy-upside-down-triangle C by getting the partial derivatives of all the parameters with
        respect to C (loss)
     */
    public void backprop(SimpleMatrix y) {
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
        //will this work: Yes!
        Layer output = this.layers.getLast();
        SimpleMatrix initalGrad = output.cachedOutput.minus(y).scale(1.0 / this.samples);
        SimpleMatrix grad = output.backwardOutput(initalGrad);
        for(Layer layer : this.layers.reversed().subList(1, this.layers.size())) {
            grad = layer.backward(grad);
        }
        //So now we need to update our weights by taking the values generated above and multiplying everything
        //this is probably dumb lol
        this.updateWeightsBiases();
    }

    public void train(int epochs, SimpleMatrix Xval, SimpleMatrix Yval) {
        for(int i = 0; i <epochs; i++) {
            SimpleMatrix y_hat = this.forwardPass(this.x);
            double loss = this.bceLoss(y_hat, this.y);
            double accuracy = this.accuracy(y_hat, this.y);
            //costs[i] = loss;
            this.backprop(this.y);
            if(((i+1) % 100) == 0) {
                System.out.println("Epoch: " + (i+1) + " Loss: " + loss);
                System.out.println("Accuracy: " + accuracy*100);
                //we don't need to save the values for our validation as we aren't
                //running the backpropagation...
                SimpleMatrix y_val = this.forwardPassNoGrad(Xval);

                double valLoss = this.bceLoss(y_val, Yval);
                double valAccuracy = this.accuracy(y_val, Yval);
                System.out.println("Validation Loss: " + valLoss + ", Validation Accuracy: " + valAccuracy*100);
            }
        }
    }

    public double predict(SimpleMatrix input) {
        assert(input.getNumCols() == 1);
        return this.forwardPass(input).get(0,0);
    }
    public void updateWeightsBiases() {
        for(Layer layer : this.layers) {
            layer.updateWeights(this.learning_rate);
            layer.updateBiases(this.learning_rate);
        }
    }
    public static boolean compare(SimpleMatrix a, SimpleMatrix b) {
        boolean isEqual = true;
        if(a.getNumCols() == b.getNumCols() && a.getNumRows() == b.getNumRows()) {
            for(int r = 0; r < a.getNumRows(); r++) {
                if(!isEqual) break;
                for(int c = 0; c < a.getNumCols(); c++) {
                    if(a.get(r, c) != b.get(r, c)) {
                        System.out.print("At index " + r + ", " + c + ": ");
                        System.out.println(a.get(r, c) + " does not equal " + b.get(r, c));
                        isEqual = false;
                        break;
                    }
                }
            }
        } else {
            System.out.println("The dimensions of a and b are different");
            isEqual = false;
        }

        return isEqual;
    }
    //todo: finish this later lol.
    public void save() {
        //placeholder for model saving... will be a json file
    }
}
