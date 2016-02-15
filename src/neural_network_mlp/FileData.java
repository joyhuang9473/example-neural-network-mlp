package neural_network_mlp;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Scanner;
import javax.swing.JFileChooser;

public class FileData {

    ArrayList<double[]> dataSet = null;

    public void setDataSet() throws FileNotFoundException {
        JFileChooser fileChooser = new JFileChooser();

        if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
            java.io.File file = fileChooser.getSelectedFile();

            Scanner input = new Scanner(file);
            dataSet = new ArrayList<double[]>();

            while (input.hasNextLine()) {
                String row_entries = input.nextLine();
                String[] entries = row_entries.split("\t");
                double[] data = new double[entries.length];

                for (int i = 0 ; i < entries.length ; i++) {
                    data[i] = Double.parseDouble(entries[i]);
                }

                dataSet.add(data);
            }
            
            input.close();
        } else {
            System.out.println("No File selected");
        }
    }
    
    public ArrayList<double[]> getDataSet() {
        return dataSet;
    }

}
