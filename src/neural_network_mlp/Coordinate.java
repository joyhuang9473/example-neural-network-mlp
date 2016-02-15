package neural_network_mlp;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.util.ArrayList;
import java.util.LinkedList;
import javax.swing.JPanel;

public class Coordinate extends JPanel {
    public static final int WIDTH = 600;
    public static final int HEIGHT = 600;

    public static final int INIT = 0;
    public static final int TYPE_BEFORE_TRAINING = 1;
    public static final int TYPE_AFTER_TRAINING = 2;
    public static final int TYPE_TESTING = 3;

    protected int type;

    protected Pattern dataSet;

    public Coordinate() {
        type = INIT;
        this.setBackground(Color.white);
    }

    public void setCollection(Pattern dataSet) {
        this.dataSet = dataSet;
    }
    
    public void setType(int type) {
        this.type = type;
    }

    public void draw() {
        repaint();
    }
    
    protected void drawSet(Graphics2D g2, ArrayList<double[]> set, Color color) {
        double[] data = null;
        Ellipse2D e;

        for (int i = 0 ; i < set.size() ; i++) {
            data = set.get(i);

            e = addPoint(data[0], data[1]);

            g2.setPaint(color);
            g2.fill(e);    
        }
    }

    protected void drawWeights(Graphics2D g2, LinkedList<Double> weights, Color color) {
        Ellipse2D e;
        double y = 0;
        
        for (int x = (-1)*WIDTH/2 ; x < WIDTH/2 ; x++) {
            y = (weights.get(0) - weights.get(1)*x) / weights.get(2);
            e = addPoint(x, y);
            g2.setPaint(color);
            g2.fill(e);
        }
    }

    protected Ellipse2D addPoint(double x, double y) {
        double point_x = WIDTH/2 + x*10;
        double point_y = HEIGHT/2 - y*10;
        
        Ellipse2D e = new Ellipse2D.Double(point_x, point_y, 3, 3);

        return e;
    }

    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        Graphics2D g2 = (Graphics2D)g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        double width = getWidth();
        double height = getHeight();
        //draw Axis
        g2.draw(new Line2D.Double(0, height/2, width, height/2));
        g2.draw(new Line2D.Double(width/2, 0, width/2, height));

        switch (type) {
            case TYPE_BEFORE_TRAINING:
                drawSet(g2, dataSet.getExamples(), Color.black);
                break;
            case TYPE_AFTER_TRAINING:
                drawSet(g2, dataSet.getTrainingExample(), Color.red);
                drawSet(g2, dataSet.getTestingExample(), Color.blue);
                break;
            default:
                break;
        }
    }

}

