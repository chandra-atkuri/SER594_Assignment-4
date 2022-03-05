package GUI;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;

import java.awt.event.*;
import java.io.File;
import java.io.IOException;

class ActionEvent implements ActionListener {
    JFrame frame=new JFrame(); //Creating the frame
    JButton button=new JButton("Run Model"); //The button to run the model

    //8 text fields to enter the body ratios
    JTextField text1 = new JTextField();
    JTextField text2 = new JTextField();
    JTextField text3 = new JTextField();
    JTextField text4 = new JTextField();
    JTextField text5 = new JTextField();
    JTextField text6 = new JTextField();
    JTextField text7 = new JTextField();
    JTextField text8 = new JTextField();

    //Labels indicating what to enter into the text field
    JLabel label1 = new JLabel();
    JLabel label2 = new JLabel();
    JLabel label3 = new JLabel();
    JLabel label4 = new JLabel();
    JLabel label5 = new JLabel();
    JLabel label6 = new JLabel();
    JLabel label7 = new JLabel();
    JLabel label8 = new JLabel();

    JLabel result1 = new JLabel(); //label to display the output

    //Labels to display the author names
    JLabel creators = new JLabel();
    JLabel author1 = new JLabel();
    JLabel author2 = new JLabel();
    JLabel author3 = new JLabel();

    ActionEvent(){
        GUI();
    }

    public void GUI(){
        //setting up the frame of the gui
        frame.setTitle("Assignment-4");
        frame.getContentPane().setLayout(null);
        frame.setVisible(true);
        frame.setBounds(200,200,400,600);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        //setting the bounds of all the elements
        creators.setBounds(10, 10, 500,25);
        creators.setText("Created By:");
        author1.setBounds(20, 35, 500, 25);
        author1.setText("Chandra Sekhar Sai Sampath Swaroop Atkuri");
        author2.setBounds(20, 60, 500, 25);
        author2.setText("Anantha Ramayya Kandrapu");
        author3.setBounds(20, 85, 500, 25);
        author3.setText("Sarthak Vats");
        label1.setBounds(70,140,100,25);
        label1.setText("Age");
        text1.setBounds(180,140,100,25);
        label2.setBounds(70,170,100,25);
        label2.setText("Weight");
        text2.setBounds(180,170,100,25);
        label3.setBounds(70,200,100,25);
        label3.setText("Height");
        text3.setBounds(180,200,100,25);
        label4.setBounds(70,230,100,25);
        label4.setText("Chest");
        text4.setBounds(180,230,100,25);
        label5.setBounds(70,260,100,25);
        label5.setText("Abdomen");
        text5.setBounds(180,260,100,25);
        label6.setBounds(70,290,100,25);
        label6.setText("Hip");
        text6.setBounds(180,290,100,25);
        label7.setBounds(70,320,100,25);
        label7.setText("Thigh");
        text7.setBounds(180,320,100,25);
        label8.setBounds(70,350,100,25);
        label8.setText("Arms");
        text8.setBounds(180,350,100,25);

        button.setBounds(130,395,100,40);

        result1.setBounds(130,450,100,40);

        //adding elements to the frame
        frame.add(creators);
        frame.add(label1);
        frame.add(label2);
        frame.add(label3);
        frame.add(label4);
        frame.add(label5);
        frame.add(label6);
        frame.add(label7);
        frame.add(label8);
        frame.add(text1);
        frame.add(text2);
        frame.add(text3);
        frame.add(text4);
        frame.add(text5);
        frame.add(text6);
        frame.add(text7);
        frame.add(text8);
        frame.add(button);
        frame.add(result1);
        frame.add(author1);
        frame.add(author2);
        frame.add(author3);
        button.addActionListener(this);
    }

    @Override
    public void actionPerformed(java.awt.event.ActionEvent e) {
        //Parsing the data from the text fields as we get text values
        Double number1 = Double.parseDouble(text1.getText());
        Double number2 = Double.parseDouble(text2.getText());
        Double number3 = Double.parseDouble(text3.getText());
        Double number4 = Double.parseDouble(text4.getText());
        Double number5 = Double.parseDouble(text5.getText());
        Double number6 = Double.parseDouble(text6.getText());
        Double number7 = Double.parseDouble(text7.getText());
        Double number8 = Double.parseDouble(text8.getText());
        double[][] bodyFatInput = {{number1, number2,
                number3, number4,
                number5, number6,
                number7, number8}};

        //Creating a MultiLayerNetwork object to load the model
        MultiLayerNetwork net2 = null;
        try {
            //Loading the model
            net2 = MultiLayerNetwork.load(new File("model"), true);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
//        double testlist[][] = {{35.0000, 187.7500,   69.5000,  101.3000,   96.4000,  100.1000,   69.0000,  36.1000}};

        /**Creating a test object with the inputted data and passing it to the model
         * to run the model on that data
         */

        INDArray tes = Nd4j.create(bodyFatInput);
        INDArray out = net2.output(tes);
        double[] ans = out.data().asDouble();
        int real_ans = 0;

        //we get the output in either 0, 1 or 2 representing the category of fat content
        for (int i = 0; i < ans.length; i++) {
            real_ans = ans[i] > ans[real_ans] ? i : real_ans;
        }
        if(real_ans == 0) {
            result1.setText("Low Body Fat");
        }
        else if (real_ans == 1){
            result1.setText("Normal Body Fat");
        }
        else {
            result1.setText("High Body Fat");
        }
    }
}

public class App {
    public static void main(String[] args)
    {
        new ActionEvent();
    }
}