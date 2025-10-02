import org.ejml.*;
import org.apache.commons.csv.*;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws FileNotFoundException {
        // Using this dataset https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
        // will at some point test with other datasets
        int[] labels = {
                1
        };
        int[] ignoreColumns = {
                0
        };
        DataProcessor data = new DataProcessor("src/main/resources/data.csv", labels, ignoreColumns);
        data.normalizeFeatures();
        data.dataSplit(0.7);
        //use colsX since we will transpose our X and Y matrices
        //this now works great!!
        List<Layer> layers = new ArrayList<>(Arrays.asList(
                new FeedForward(data.colsX, 16, new Relu(), "Layer 1"),
                new FeedForward(16, 8, new Relu(), "Layer 2"),
                new FeedForward(8, 1, new Sigmoid(), "Output layer")
        ));


        Network arch = new Network(layers, data.Xtrain.transpose(), data.Ytrain.transpose(), 0.01);
        int epochs = 4000;
        //arch.save();
        //double[] costs = new double[epochs];
        //TODO move calculations to GPU...
        //TODO we can also do multiple epochs at once like pytorch does
        arch.train(epochs, data.Xval.transpose(), data.Yval.transpose());

        double result = data.Yval.transpose().getColumn(0).get(0);
        //converting the networks prediction into a format matching the label
        //is up to the user. I can't think of any way to automatically convert this
        double prediction = arch.predict(data.Xval.transpose().getColumn(0)) > 0.5 ? 1 : 0;
        //since we encode the labels as numerical values, our dataprocessor should be able to
        //decode those values back into the labels
        System.out.println("Our network predicted: " + prediction + " and the actual result is " + result);
        if(prediction == result) {
            System.out.println("Our network is correct!");
        } else {
            System.out.println("Our network got this one wrong :(");
        }


    }
}