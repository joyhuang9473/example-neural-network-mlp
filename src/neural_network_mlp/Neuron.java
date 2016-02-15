package neural_network_mlp;

import java.util.Random;

public class Neuron {

    private double[] relativeWeights;
    private double activation;

    public Neuron(int pre_num_neurons) {
        relativeWeights = new double[pre_num_neurons];
        Random random = new Random();

        /**
         * Initialize weights with random values between interval [-0.5,0.5]
         */
        for (int i = 0 ; i < pre_num_neurons ; i++) {
            relativeWeights[i] = random.nextDouble() - 0.5;
        }
    }

    public double activate(double inputs[]) throws Exception {
        if (inputs.length != relativeWeights.length) throw new Exception();

        activation = 0;

        for (int i = 0 ; i < inputs.length ; i++) {
            activation += inputs[i] * relativeWeights[i];
        }

        activation = (double)(1 / (1 + Math.exp((-1)*activation)));
        
        return activation;
    }

    public double activationDerivative() {
        return activation * (1 - activation);
    }
    
    public double[] getRelativeWeights() {
        return relativeWeights;
    }

    public double getRelativeWeight(int i) {
        return relativeWeights[i];
    }

    public void setRelativeWeight(int i, double value) {
        relativeWeights[i] = value;
    }

}
