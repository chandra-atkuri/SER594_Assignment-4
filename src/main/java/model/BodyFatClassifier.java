package model;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;

public final class BodyFatClassifier {
    private BodyFatClassifier() {
    }

    private static Logger log = LoggerFactory.getLogger(BodyFatClassifier.class);

    public static void main(String[] args) throws IOException, InterruptedException {

        try (RecordReader recordReader = new CSVRecordReader(1, ',')) {
            //reading the data file
            recordReader.initialize(new FileSplit(new File("bfwc.csv")));
            //setting up the iterator with its parameters
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,32, 8, 3);
            DataSet allData = iterator.next();
            //shuffling the data
            allData.shuffle(42);

            //Splitting the data into train and test
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.7);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();

            // DataNormalization normalizer = new NormalizerStandardize();
            // normalizer.fit(trainingData);
            // normalizer.transform(trainingData);
            // normalizer.transform(testData);

            final int numInputs = 8;
            int numOutputs = 3;
            int numHiddenNodes = 6;
            int seed = 123;
            double learningRate = 0.01;

            //Setting up the model
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .weightInit(WeightInit.UNIFORM)
                    .updater(new Nesterovs(learningRate, 0.9))
                    .list()
                    .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes-1)
                            .activation(Activation.SIGMOID)
                            .build())
                    .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes-2)
                            .activation(Activation.TANH)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunction.MSE)
                            .activation(Activation.SOFTMAX)
                            .nIn(numHiddenNodes-1).nOut(numOutputs).build())
                    .build();

            // run the model
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            model.setListeners(new ScoreIterationListener(100));
            for (int i = 0; i < 11800; i++) {
                model.fit(trainingData);
            }
            Evaluation eval = new Evaluation();
            INDArray output = model.output(testData.getFeatures());
            eval.eval(testData.getLabels(), output);
            // System.out.println(testData);
            System.out.println(output);
            System.out.println(eval.stats());
            if (eval.accuracy() >= 0.75)
                model.save(new File("model"), false);
        }

    }
}