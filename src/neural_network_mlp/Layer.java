package neural_network_mlp;

import java.util.ArrayList;

public class Layer {

    private int prev_num_neurons, num_neurons;
    private ArrayList<Neuron> neurons;
    private double outputs[];

    public Layer(int prev_num_neurons, int num_neurons) {
        this.prev_num_neurons = prev_num_neurons + 1; // add bias
        this.num_neurons = num_neurons + 1; // add bias

        neurons = new ArrayList<Neuron>();
        outputs = new double[this.num_neurons];

        for (int i = 0 ; i < this.num_neurons ; i++) {
            neurons.add(new Neuron(this.prev_num_neurons));
        }
    }

    public static double[] addBias(double[] in) {
        double out[] = new double[in.length + 1];

        for (int i = 0 ; i < in.length ; i++) {
            out[i+1] = in[i];
        }

        out[0] = -1;

        return out;
    }

    public static double[] removeBias(double[] in) {
        double out[] = new double[in.length - 1];

        for (int i = 0 ; i < in.length-1 ; i++) {
            out[i] = in[i+1];
        }

        return out;
    }
    
    public double[] evaluate(double in[]) throws Exception {
        double inputs[];

        inputs = (in.length != getWeights(0).length) ? addBias(in) : in;

        if (inputs.length != getWeights(0).length) throw new Exception();

        for (int i = 1 ; i < num_neurons ; i++) {
            outputs[i] = neurons.get(i).activate(inputs);
        }

        //bias
        outputs[0] = -1;

        return outputs;
    }
    
    public int getSize() {
        return num_neurons;
    }

    public void setOutput(double[] outputs) {
        this.outputs = outputs;
    }

    public double getOutput(int j) {
        return outputs[j];
    }

    public double getActivationDerivative(int j) {
        return neurons.get(j).activationDerivative();
    }
    
    public double[] getWeights(int j) {
        return neurons.get(j).getRelativeWeights();
    }

    public double getWeight(int i, int j) {
        return neurons.get(j).getRelativeWeight(i);
    }
    
    public void setWeight(int i, int j, double value) {
        neurons.get(j).setRelativeWeight(i, value);
    }

}
