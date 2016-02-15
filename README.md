![breast-cancer-wisconsin@480.png](http://imgur.com/mNbghuO.png?1)

Example-Neural-Network-MLP
=========

Example-Neural-Network-MLP is an implementation of multilayer perceptron (MLP), which is a feedforward artificial neural network model, written in Java.

Junior project in class "Neural Network". Last updated 11/05/2014.

How to use
-----------------------

- Select input file from `dataSet` directory.
- Field `學習率 (learning rate)` and `學習次數 (count of learning loop)` columns
or Field blank to use the default value.
- Click `新增隱藏層` button to create a hidden layer and field column to set the number of neurons in the hidden layer.
- Click `訓練` button to train.
- Click `測驗` button to test.

Feature
-----------------------

### Graphics ###

Blue dots for testing set.

Red dots for training set.

### Console log ###

The leftmost field in gui is console log output.

It will show correct rate and RMSE(Root-Mean-Square Error) in the progress of training or testing.

    // console log example
    Training
    =========
    Training Correct Rate: 1.0
    RMSE: 0.0

    Testing
    =========
    Testing Correct Rate: 1.0
    RMSE: 0.0

### DataSet (Trainging Set, Testing Set) ###

    // src/neural_netowrk_mlp/Pattern.java
    int num_training = 2*dataSize/3 + 1;
    int num_testing = dataSize - num_training;

### Rate ###

correctRateForTraining, correctRateForTesting, RMSE

    // src/neural_netowrk_mlp/MLP.java
    ...
    public void train(ArrayList<double[]> examples, ArrayList<double[]> expectResults) {
        ...
        try {
            correctRateForTraining = getCorrectRate(batch_net_outputs, expectResults);
            console_message += "Training Correct Rate: " + correctRateForTraining + "\n";
            console_message += "RMSE: " + Math.sqrt(evaluateQuadraticError(batch_net_outputs, expectResults)) + "\n\n";

            consoleOutput.setText(console_message);
            System.out.println(console_message);
        } catch (Exception e) {
            ...

    // src/neural_netowrk_mlp/MLP.java
    ...
    public void test(ArrayList<double[]> examples, ArrayList<double[]> expectResults) {
        ...
        try {
            correctRateForTesting = getCorrectRate(batch_net_outputs, expectResults);
            console_message += "Testing Correct Rate: " + correctRateForTesting + "\n";
            console_message += "RMSE: " + Math.sqrt(evaluateQuadraticError(batch_net_outputs, expectResults)) + "\n\n";

            consoleOutput.setText(console_message);
            System.out.println(console_message);
        } catch (Exception e) {
            ...

### Default Setting ###

    // src/neural_netowrk_mlp/Framework.java
    int learningCount = (!jt_learningCount.getText().isEmpty()) ? Integer.parseInt(jt_learningCount.getText()) : 10000;
    double learningRate = (!jt_learningRate.getText().isEmpty()) ? Double.parseDouble(jt_learningRate.getText()) : 0.5;

    // src/neural_netowrk_mlp/Neuron.java
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

### Back Propagation ###

    // src/neural_netowrk_mlp/MLP.java
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

Dependency
-----------------------

JDK

    $ java -version

    java version "1.8.0_66"
    Java(TM) SE Runtime Environment (build 1.8.0_66-b17)
    Java HotSpot(TM) 64-Bit Server VM (build 25.66-b17, mixed mode)

Build and run
-----------------------

### Compile ###

    $ javac -d bin -sourcepath src src/neural_network_mlp/*.java

### Run ###

    $ java -cp bin neural_network_mlp.Main

### Create jar file ###

    $ jar cfe neural_network_mlp.jar neural_network_mlp.Main -C bin/ .

    $ jar tf neural_network_mlp.jar # list table of contents for archive

    META-INF/
    META-INF/MANIFEST.MF
    .gitkeep
    neural_network_mlp/
    neural_network_mlp/Coordinate.class
    neural_network_mlp/FileData.class
    neural_network_mlp/Framework$CustomActionListener.class
    neural_network_mlp/Framework.class
    neural_network_mlp/Layer.class
    neural_network_mlp/Main.class
    neural_network_mlp/MLP.class
    neural_network_mlp/Neuron.class
    neural_network_mlp/Pattern.class

Screenshot
-----------------------

![xor@480.png](http://imgur.com/3QC7zrE.png?1)

![579@480.png](http://imgur.com/1UI3gnJ.png?1)

![wine@480.png](http://imgur.com/knZRdYj.png?1)

More details in [http://imgur.com/a/2pr07](http://imgur.com/a/2pr07)

Reference
-----------------------

[1]. Sergiy Kovalchuk, "[How to Compile and Run Java Code from a Command Line](http://www.sergiy.ca/how-to-compile-and-launch-java-code-from-command-line/)", 2011

[2]. StackOverFlow, "[Create jar file from command line](http://stackoverflow.com/questions/11243442/create-jar-file-from-command-line)", 2012
