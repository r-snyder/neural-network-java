import java.util.Arrays;
import linearalgebra.*;
//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    /*
    So...each layer is composed of nodes
    Here are the rules of neural networks you should keep in mind:

    For a node in a layer, it’s connected to every single node in the next layer
    through connections called weights.
    Weights are randomly initialized to begin with, but we change them based on the gradient ∇C.
    Every node in the network have a bias,
    which is added to the node after the weight-node multiplication.
    The initial initialValues for the biases don’t matter — just assign them randomly.
    The value of the node (excluding input) is
    the dot product of each of the previous layers nodes, and their weights
    As an example, lets use the numbers from the medium article. - DONE
     */
    public static void main(String[] args) {
        // using linearalgebra package from https://github.com/danhales/linearalgebra/tree/main
        // will eventually switch to EJML
        int[] network = {2, 3, 3, 1};
        // should read this from a file, like a csv.
        double[][] x = {{
            150d, 70d
        }, {
            254d, 73d
        }, {
            312d, 68d
        }, {
            120d, 60d
        }, {
            154d, 61d
        }, {
            212d, 65d
        }, {
            216d, 67d
        }, {
            145d, 67d
        }, {
            184d, 64d
        }, {
            130d, 69d
        }};
        double[][] y = {{
            0
        }, {
            1
        }, {
            1
        }, {
            0
        }, {
            0
        }, {
            1
        }, {
            1
        }, {
            0
        }, {
            1
        }, {
            0
        }};
        // our input matrix
        Matrix X = new Matrix(x).transpose();
        Matrix Y = new Matrix(y).transpose();
        Network arch = new Network(network, X.getNumColumns(), 0.01);
        //print out weight and biases shapes...
        System.out.println(arch.layers.length);
        //we have an issue with our starting weights being two large for sigmoid but its fine for now
        int m = y.length;
        //ok time to do a first test of 10 epochs with printing out the losses...
        int epochs = 500;
        //double[] costs = new double[epochs];
        //TODO move calculations to GPU...
        for(int i = 0; i <epochs; i++) {
            Matrix y_hat = arch.forwardPass(X);
            double loss = arch.bceLoss(y_hat, Y);
            //costs[i] = loss;
            arch.testBackprop(m, Y, X);
            if(((i+1) % 20) == 0) System.out.println("Epoch: " + (i+1) + " Loss: " + loss);
        }

    }
}