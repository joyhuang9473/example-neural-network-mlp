package neural_network_mlp;

import java.util.ArrayList;
import javax.swing.JFrame;

public class Main {

    public static Framework frame = new Framework();

    public static void main(String[] args) {
        frame.setSize(1000, 650);
        frame.setLocationRelativeTo(null);
        frame.setTitle("Neural Network MLP");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

}
