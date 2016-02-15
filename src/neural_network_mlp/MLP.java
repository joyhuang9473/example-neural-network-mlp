package neural_network_mlp;

import java.util.ArrayList;
import javax.swing.JTextArea;

public class MLP {

    private ArrayList<Layer> layers;
    private ArrayList<double[][]> weightDelta;
    private ArrayList<double[]> gradent;
    private double learningRate;
    private double correctRateForTraining;
    private double correctRateForTesting;
    private JTextArea consoleOutput;
    private String console_message = "";
    
    public MLP(int[] num_layers, JTextArea consoleOutput) {
        layers = new ArrayList<Layer>();
        weightDelta = new ArrayList<double[][]>();
        gradent = new ArrayList<double[]>();

        this.consoleOutput = consoleOutput;

        for (int i = 0 ; i < num_layers.length; i++) {
            layers.add(
                    new Layer(
                        i == 0 ? num_layers[i] : num_layers[i-1],
                        num_layers[i]
                    )
            );
            weightDelta.add(new double[layers.get(i).getSize()][layers.get(i).getWeights(0).length]);
            gradent.add(new double[layers.get(i).getSize()]);
        }
    }

    public double[] passNet(double[] inputs) {
        // pass the inputs through all neural network
        double[] outputs = null;

        for (int i = 0 ; i < layers.size() ; i++) {
            try {
                if (i == 0) { // input layer
                    inputs = Layer.addBias(inputs);
                    layers.get(i).setOutput(inputs);
                    continue;
                }

                outputs = layers.get(i).evaluate(inputs);
                inputs = outputs;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return outputs;
    }

    public double evaluateError(double[] net_output, double[] desired_output) throws Exception {
        desired_output = (desired_output.length != net_output.length) ? Layer.addBias(desired_output) : desired_output;

        if (net_output.length != desired_output.length) throw new Exception();
        
        double sum_e = 0;
        
        for (int i = 1 ; i < net_output.length ; i++) {
            sum_e += (desired_output[i] - net_output[i])*(desired_output[i] - net_output[i]);
        }

        // E(n)
        return (double)(sum_e/2);
    }

    public double evaluateQuadraticError(ArrayList<double[]> net_outputs, ArrayList<double[]> desired_outputs) throws Exception {
        double sum_e = 0;

        for (int i = 0 ; i < net_outputs.size() ; i++) {
            sum_e += evaluateError(net_outputs.get(i), desired_outputs.get(i));
        }

        //Eav
        return (double)(sum_e/net_outputs.size());
    }

    public void evaluateGradients(double[] expectResults) {
        expectResults = Layer.addBias(expectResults); // add bias

        for (int l = layers.size()-1 ; l > 0 ; l--) {
            for (int j = 1 ; j < layers.get(l).getSize() ; j++) {
                if (l == layers.size()-1) {
                    gradent.get(l)[j] = (expectResults[j] - layers.get(l).getOutput(j)) * layers.get(l).getActivationDerivative(j);
                } else {
                    double sum = 0;

                    for (int k = 1 ; k < layers.get(l+1).getSize() ; k++) {                        
                        sum += gradent.get(l+1)[k] * layers.get(l+1).getWeight(j, k);
                    }

                    gradent.get(l)[j] = layers.get(l).getActivationDerivative(j) * sum;
                }
            }
        }
    }

    public void resetWeightsDelta() {
        for (int l = 0 ; l < layers.size() ; l++) {
            for (int j = 0 ; j < layers.get(l).getSize() ; j++) {
                double[] weights = layers.get(l).getWeights(j);

                for (int i = 0 ; i < weights.length ; i++) {
                    weightDelta.get(l)[j][i] = 0;
                }
            }
        }
    }

    public void evaluateWeightsDelta() {
        for (int l = 1 ; l < layers.size() ; l++) {
            for (int j = 1 ; j < layers.get(l).getSize() ; j++) {
                double [] weights = layers.get(l).getWeights(j);

                for (int i = 0 ; i < weights.length ; i++) {
                    weightDelta.get(l)[j][i] = learningRate*gradent.get(l)[j]*layers.get(l-1).getOutput(i);
                }
            }
        }
    }

    public void updateWeights() {
        for (int l = 1 ; l < layers.size() ; l++) {
            for (int j = 1 ; j < layers.get(l).getSize() ; j++) {
                double[] weights = layers.get(l).getWeights(j);
                
                for (int i = 0 ; i < weights.length ; i++) {
                    layers.get(l).setWeight(i, j, weights[i]+weightDelta.get(l)[j][i]);
                }
            }
        }
    }

    private ArrayList<double[]> batchBackPropagation(ArrayList<double[]> examples, ArrayList<double[]> expectResults) {
        resetWeightsDelta();

        ArrayList<double[]> batch_net_outputs = new ArrayList<double[]>();

        for (int i = 0 ; i < examples.size() ; i++) {
            double[] net_outputs = passNet(examples.get(i));
            net_outputs = Layer.removeBias(net_outputs);
            batch_net_outputs.add(net_outputs);

            evaluateGradients(expectResults.get(i));
            evaluateWeightsDelta();
            updateWeights();
        }

        return batch_net_outputs;
    }

    public void train(ArrayList<double[]> examples, ArrayList<double[]> expectResults) {
        if (console_message.length() > 1000) console_message = "";

        console_message += "Training\n";
        console_message += "=========\n";

        ArrayList<double[]> batch_net_outputs = null;

        batch_net_outputs = batchBackPropagation(examples, expectResults);

        try {
            correctRateForTraining = getCorrectRate(batch_net_outputs, expectResults);
            console_message += "Training Correct Rate: " + correctRateForTraining + "\n";
            console_message += "RMSE: " + Math.sqrt(evaluateQuadraticError(batch_net_outputs, expectResults)) + "\n\n";

            consoleOutput.setText(console_message);
            System.out.println(console_message);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void test(ArrayList<double[]> examples, ArrayList<double[]> expectResults) {
        if (console_message.length() > 1000) console_message = "";

        console_message += "Testing\n";
        console_message += "=========\n";

        ArrayList<double[]> batch_net_outputs = new ArrayList<double[]>();

        for (int i = 0 ; i < examples.size() ; i++) {
            double[] net_outputs = passNet(examples.get(i));
            net_outputs = Layer.removeBias(net_outputs);
            batch_net_outputs.add(net_outputs);
        }

        try {
            correctRateForTesting = getCorrectRate(batch_net_outputs, expectResults);
            console_message += "Testing Correct Rate: " + correctRateForTesting + "\n";
            console_message += "RMSE: " + Math.sqrt(evaluateQuadraticError(batch_net_outputs, expectResults)) + "\n\n";

            consoleOutput.setText(console_message);
            System.out.println(console_message);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getCorrectRate(ArrayList<double[]> net_outputs, ArrayList<double[]> expect_outputs) throws Exception {
        int num_success = 0;
        int num_total = expect_outputs.size();

        for (int i = 0 ; i < expect_outputs.size() ; i++) {
            double[] expect_output = expect_outputs.get(i);
            double[] net_output = net_outputs.get(i);

            if (net_output.length != expect_output.length) throw new Exception();
            
            boolean flag_identicle = true;

            for (int j = 0 ; j < expect_output.length ; j++) {
                double advised_value = (net_output[j] >= 0.5) ? 1 : 0;

                System.out.println("expect output: " +  expect_output[j]);
                System.out.println("net output: " +  net_output[j]);
                System.out.println("advised_value outputs: " + advised_value + "\n");

                if (advised_value != expect_output[j]) {
                    flag_identicle = false;
                }
            }
            
            if (flag_identicle) {
                ++num_success;
            }
        }

        return (double)num_success/(double)num_total;
    }
}
