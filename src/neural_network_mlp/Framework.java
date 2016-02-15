package neural_network_mlp;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;

public class Framework extends JFrame {

    private MLP mlp;
    private FileData read_file = new FileData();
    private Coordinate graphPanel = new Coordinate();
    private Pattern dataPattern;
    
    // the setting of GUI
    public JPanel inputPanel = null;
    public JTextField jt_learningRate = new JTextField(8);
    public JTextField jt_learningCount = new JTextField(8);
    public ArrayList<JTextField> jt_hiddenlayers = new ArrayList<JTextField>();

    public JButton jb_select_file = new JButton("選擇檔案");
    public JButton jb_add_hidden_layer = new JButton("新增隱藏層");
    public JButton jb_training = new JButton("訓練");
    public JButton jb_testing = new JButton("測驗");

    public JPanel resultPanel = null;
    public JTextArea consoleOutput = null;

    public Framework() {
        setLayout(null);

        inputPanel = setInputPanel();
        resultPanel = setResultPanel();

        resultPanel.setBounds(0, 0, 200, 600);
        graphPanel.setBounds(200, 0, 600, 600);
        inputPanel.setBounds(800, 0, 200, 600);
        
        jb_select_file.addActionListener(new CustomActionListener());
        jb_add_hidden_layer.addActionListener(new CustomActionListener());
        jb_training.addActionListener(new CustomActionListener());
        jb_testing.addActionListener(new CustomActionListener());

        add(resultPanel);    
        add(graphPanel);
        add(inputPanel);
    }

    protected JPanel setResultPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BorderLayout());
        
        consoleOutput = new JTextArea(5, 20);
        consoleOutput.setMargin(new Insets(5, 5, 5, 5));
        consoleOutput.setEditable(false);
        panel.add(new JScrollPane(consoleOutput), BorderLayout.CENTER);

        return panel;    
    }

    protected JPanel setInputPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new FlowLayout(FlowLayout.LEFT, 5, 20));

        panel.add(jb_select_file);
        panel.add(new JLabel("Input file"));
        panel.add(new JLabel("學習率:"));
        panel.add(jt_learningRate);
        panel.add(new JLabel("學習次數:"));
        panel.add(jt_learningCount);
        panel.add(new JLabel("(欄位空白為套用預設值)"));
        panel.add(jb_add_hidden_layer);
        panel.add(jb_training);
        panel.add(jb_testing);

        return panel;
    }
    
    class CustomActionListener implements ActionListener {
        public void actionPerformed(ActionEvent e) {
            if (e.getSource() == jb_select_file) {
                try {
                    read_file.setDataSet();
                    dataPattern = new Pattern();
                    dataPattern.set(read_file.getDataSet());
    
                    graphPanel.setCollection(dataPattern);
                    graphPanel.setType(Coordinate.TYPE_BEFORE_TRAINING);
                    graphPanel.draw();
                } catch (FileNotFoundException e1) {
                    e1.printStackTrace();
                }
            } else if (e.getSource() == jb_training) {
                if (read_file == null) {
                    JOptionPane.showMessageDialog(null, "請選擇檔案");
                    return;
                }

                int input_layer_neurons = dataPattern.getExamples().get(0).length;
                int output_layer_neurons = dataPattern.getResults().get(0).length;
                int[] nn_neurons = {input_layer_neurons, 2, output_layer_neurons};
                
                if (jt_hiddenlayers.size() > 0) {
                    nn_neurons = new int[jt_hiddenlayers.size()+2];

                    // input layer
                    nn_neurons[0] = input_layer_neurons;

                    // hidden layer
                    for (int i = 0 ; i < jt_hiddenlayers.size() ; i++) {
                        nn_neurons[i+1] = Integer.parseInt(jt_hiddenlayers.get(i).getText());
                    }

                    // output layer
                    nn_neurons[nn_neurons.length-1] = output_layer_neurons;
                }
                
                int learningCount = (!jt_learningCount.getText().isEmpty()) ? Integer.parseInt(jt_learningCount.getText()) : 10000;
                double learningRate = (!jt_learningRate.getText().isEmpty()) ? Double.parseDouble(jt_learningRate.getText()) : 0.5;
                
                mlp = new MLP(nn_neurons, consoleOutput);
                mlp.setLearningRate(learningRate);

                try {
                    for (int i = 0 ; i < learningCount ; i++) {
                        mlp.train(dataPattern.getTrainingExample(), dataPattern.getTrainingResult());
                    }
                } catch(Exception e1) {
                    e1.printStackTrace();
                }

                graphPanel.setType(Coordinate.TYPE_AFTER_TRAINING);
                graphPanel.draw();
            } else if (e.getSource() == jb_testing) {
                mlp.test(dataPattern.getTestingExample(), dataPattern.getTestingResult());
            } else if (e.getSource() == jb_add_hidden_layer) {
                JTextField field_hidden_layer = new JTextField(8);
                jt_hiddenlayers.add(field_hidden_layer);
                
                inputPanel.add(new JLabel("\n隱藏層" + jt_hiddenlayers.size() + " 神經元個數:"));
                inputPanel.add(field_hidden_layer);
                inputPanel.revalidate();
            }
        }
    }
}
