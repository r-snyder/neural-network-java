import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.ejml.simple.SimpleMatrix;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/*
    Class for handling data processing tasks...
    Given the label column, and any columns that we should ignore,
    this class will automatically convert the label to numerical targets and
    process the features
    Also has an option to normalize the features, if needed.
 */
public class DataProcessor {
    //the minimum and maximum values for each feature (a.k.a column)
    private double[] featureMins, featureMaxes;
    public int rowsX;
    public int colsX;
    //our data always needs an X and a Y...
    //for now we will assume that they come from the same file
    public SimpleMatrix X;
    public SimpleMatrix Y;
    //for now we will also keep the original matrices
    //and also make two new matrices of each for training and validation..
    public SimpleMatrix Xtrain;
    public SimpleMatrix Xval;
    public SimpleMatrix Ytrain;
    public SimpleMatrix Yval;

    public Map<String, Map<String, Integer>> labelMapped;

    public DataProcessor(String file, int[] labels, int[] ignoreColumns) throws FileNotFoundException {
        try {
            Reader in = new FileReader(file);
            CSVParser parser = CSVFormat.EXCEL.builder().setHeader().get().parse(in);
            List<CSVRecord> records = parser.getRecords();
            this.rowsX = records.size();
            // I don't know why, but https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data says we have
            // 32 columns, but records.get(0).size() returns 33
            // so we simply add 1 to labels.length + ignoreColumns.length
            // nevermind, if we setHeaders it gets fixed lol
            this.colsX = records.getFirst().size() - (labels.length + ignoreColumns.length);
            //we must pass in the index of the label(s) column
            //and what columns to ignore
            //the size of our X matrix will be, this.cols - labels.length - ignoreColumns.length
            this.X = new SimpleMatrix(this.rowsX, this.colsX);
            this.Y = new SimpleMatrix(this.rowsX, 1);
            this.processFeaturesAndLabels(records, labels, ignoreColumns);


        } catch(FileNotFoundException e) {
            throw new FileNotFoundException("We couldn't find a file at that location. Please double check where your file is located");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    private void processFeaturesAndLabels(List<CSVRecord> records, int[] labels, int[] ignoreColumns) {
        // process the features from our file...
        // what happens if a column is of type string?
        // technically all record values are strings...but most we just convert to doubles
        AtomicInteger count = new AtomicInteger(0);
        /*
            For our labels...since we're going through every record (aka row) anyways
            we will do it in the same function
            So for our labels...we need to check if they can be converted to doubles or not
            if they can, then great
            if not (which is the case for our current example),
            then we need to try to one-hot encode them.
            we can do this by getting each unique value and encoding them
            or....we can get the user to do it lol
            as a proof of concept, I will do it but my only worry is that we won't have any control
            over which is encoded as which. For example, in our current case of binary classification for breast cancer,
            where 0 is false and 1 is true, then it should be B (benign) = 0 and M (malignant) = 1
         */
        // we know what columns our labels are...
        // so we want to make a new CSVRecord list of only those columns...
        Map<Integer, List<String>> labelColumns = Arrays.stream(labels).boxed().collect(Collectors.toMap(
                columnIndex -> columnIndex,
                columnIndex -> records.stream().filter(r -> r.getRecordNumber() != 1)
                        .map(record -> record.get(columnIndex))
                        .distinct().collect(Collectors.toList())

        ));
        //now that we have each label column we need to create a mapping of values where we take the original value
        //and map it to an int/double
        //in order to get the header names, we need to get the parser of the record...
        Map<String, Map<String, Integer>> labelMapping = labelColumns.entrySet().stream().collect(
                Collectors.toMap(
                      key -> records.getFirst().getParser().getHeaderNames().get(key.getKey()),
                        entry -> IntStream.range(0, entry.getValue().size())
                                .boxed()
                                .collect(Collectors.toMap(
                                        i -> entry.getValue().get(i),  // distinct value
                                        i -> i                          // sequential int
                                ))
                )
        );

        this.labelMapped = labelMapping;
        for(CSVRecord record: records) {
            //never mind, we don't need this anymore lol
            //if (hasHeaderRow && record.getRecordNumber() == 1) continue;
            //this is probably overcomplicated but its my first time
            //using streams so I just wanted to try
            List<String> features = IntStream.range(0, record.size())
                    .filter(i -> Arrays.stream(labels).noneMatch(col -> col == i)
                            && Arrays.stream(ignoreColumns).noneMatch(col -> col == i))
                    .mapToObj(record::get)
                    .toList();
            List<Double> labelList = labelMapping.entrySet().stream().map(entry ->
                    (double) entry.getValue().get(record.get(entry.getKey()))).toList();
            //now we need to turn it into an array of doubles and put it into the first row of our Matrix...
            double[] processedFeatures = features.stream().mapToDouble(Double::parseDouble).toArray();
            //is this really the only way to turn a Double list into a double array?
            //probably not lol
            double[] processedLabels = labelList.stream().mapToDouble(d -> d).toArray();

            //Arrays.stream(processedFeatures).sequential().forEach(System.out::println);
            // using AtomicInteger here because if we have a header row
            // we need to do getRecordNumber - 2 (since recordNumber starts at 1 and goes to records.size())
            // but if we don't have a header row it would getRecordNumber - 1
            this.X.setRow(count.get(), 0, processedFeatures);
            this.Y.setRow(count.getAndIncrement(), 0, processedLabels);
        }
    }

    //if needed normalize our features...
    public void normalizeFeatures() {
        System.out.println("Normalizing features...");
        System.out.println("Magnitude of features before: " + this.X.normF());
        //is our data normalized?
        boolean isNormalized = true;
        //now that we aren't dealing with a transposed matrix we can just directly operate
        //on the columns..
        //don't love this code tbh...feels like we could do it better.
        //mainly instantiating the two below variables
        //The reasoning for storing the featureMaxes and the featureMinimums is so
        //we can properly scale data that a user may provide us with.
        double featureMax;
        double featureMin;
        this.featureMaxes = new double[this.colsX];
        this.featureMins = new double[this.colsX];
        for(int col = 0; col < this.colsX; col++) {
            SimpleMatrix feature = this.X.getColumn(col);
            featureMax = feature.elementMax();
            featureMin = feature.elementMin();
            this.featureMaxes[col] = featureMax;
            this.featureMins[col] = featureMin;
            SimpleMatrix scaledFeature = feature.minus(featureMin).divide(featureMax - featureMin);
            this.X.setColumn(col, scaledFeature);
        }
        System.out.println("Finished normalizing features. feature magnitude is now " + this.X.normF());
    }

    //method to generate a training/validation split
    //so this will basically take our X and our Y
    //and split it into two matrices, one training and one validation
    //how do we want to store these?
    public void dataSplit(double splitPercent) {
        //must be less than one but greater than 0
        assert(splitPercent < 1.0d && splitPercent > 0d);
        //we need to take the total number of rows and calculate the percentage
        int trainingRows = Math.toIntExact(Math.round(this.rowsX * splitPercent));
        System.out.println("Taking " + trainingRows + " out of " + this.rowsX + " for training");
        //we should shuffle our rows...
        //but we won't for now...
        this.Xtrain = this.X.rows(0, trainingRows);
        this.Xval = this.X.rows(trainingRows, this.rowsX);
        this.Ytrain = this.Y.rows(0, trainingRows);
        this.Yval = this.Y.rows(trainingRows, this.rowsX);
    }
    //add code to get some random samples from our dataset for testing predictions on...
    //as well add some code to take the networks prediction and return the actual label... (this is maybe unnecessary)

}
