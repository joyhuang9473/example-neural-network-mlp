package neural_network_mlp;

import java.util.ArrayList;
import java.util.LinkedList;

public class Pattern {

    private ArrayList<double[]> examples = new ArrayList<double[]>();
    private ArrayList<double[]> results = new ArrayList<double[]>();

    private LinkedList<Double> clusters = new LinkedList<Double>();
    private ArrayList<double[]> binaryMapping = new ArrayList<double[]>();

    private ArrayList<double[]> training_examples = new ArrayList<double[]>();
    private ArrayList<double[]> training_results = new ArrayList<double[]>();
    private ArrayList<double[]> testing_examples = new ArrayList<double[]>();
    private ArrayList<double[]> testing_results = new ArrayList<double[]>();

    public void set(ArrayList<double[]> dataSet) {
        for (int i = 0 ; i < dataSet.size() ; i++) {
            double[] data = dataSet.get(i);
            double[] example = new double[data.length-1];
            double[] result = new double[1];

            for (int j = 0 ; j < data.length-1 ; j++) {
                example[j] = data[j];
            }
            
            result[0] = data[data.length-1];
            
            if (!clusters.contains(result[0])) clusters.add(result[0]);
            
            examples.add(example);
            results.add(result);
        }

        createBinaryResult();
        transformExpectClassToBinaryResult();
        divideTraingingAndTesting(dataSet.size());
    }

    protected void divideTraingingAndTesting(int dataSize) {
        // less input
        if (dataSize < 10) {
            for ( int i = 0 ; i < dataSize ; i++ ) {
                training_examples.add(examples.get(i));
                training_results.add(results.get(i));
                testing_examples.add(examples.get(i));
                testing_results.add(results.get(i));
            }
            
            return;
        }

        int num_training = 2*dataSize/3 + 1;
        int num_testing = dataSize - num_training;

        for (int i = 0 ; i < num_training ; i++) {
            training_examples.add(examples.get(i));
            training_results.add(results.get(i));
        }

        for (int j = 0 ; j < num_testing ; j++) {
            testing_examples.add(examples.get(num_training+j));
            testing_results.add(results.get(num_training+j));
        }    
    }
    
    protected void createBinaryResult() {
        int num_clusters = clusters.size();
        int sum_digit = 0;
        double[] digits = null;

        while (Math.pow(2, sum_digit) < num_clusters) {
            ++sum_digit;
        }

        digits = new double[sum_digit];

        for (int i = 0 ; i < sum_digit ; i++) {            
            digits[i] = 0;
        }

        for (int i = 0 ; i < clusters.size() ; i++) {
            if (digits[sum_digit-1] == 2) {
                digits[sum_digit-1] = 0;
                ++digits[sum_digit-2];

                int index_digit = sum_digit-2;

                while (index_digit >=0 && digits[index_digit] == 2) {
                    digits[index_digit] = 0;
                    ++digits[index_digit-1];
                    --index_digit;
                }
            }

            binaryMapping.add(digits.clone());            
            ++digits[sum_digit-1];
        }
    }

    protected void transformExpectClassToBinaryResult() {
        for (int i = 0 ; i < results.size() ; i++) {
            results.set(i, mapClusterToBinaryResult(results.get(i)[0]));
        }
    }

    protected double[] mapClusterToBinaryResult(double cluster) {
        return binaryMapping.get(clusters.indexOf(cluster));
    }

    public ArrayList<double[]> getExamples() {
        return examples;
    }

    public ArrayList<double[]> getResults() {
        return results;
    }

    public ArrayList<double[]> getTrainingExample() {
        return training_examples;
    }
    
    public ArrayList<double[]> getTrainingResult() {
        return training_results;
    }

    public ArrayList<double[]> getTestingExample() {
        return testing_examples;
    }
    
    public ArrayList<double[]> getTestingResult() {
        return testing_results;
    }

}
