
import java.io.File;
import java.util.Arrays;

public class Mnist {
    public static double[] sample_input;
    public static double [] sample_correct_output;


    public static void main(String[] args) {
        int inputLayer = 28 * 28;
        int firstHiddenLayer = 70;
        int secondHiddenLayer = 35;
        int outputLayer = 10;
        Network network = new Network(inputLayer, firstHiddenLayer, secondHiddenLayer, 15, outputLayer);
        TrainSet set = createTrainSet(0, 4999);
        trainData(network, set, 100, 50, 100);

        TrainSet testSet = createTrainSet(5000, 9999);
        testTrainSet(network, testSet, 10);

        System.out.println("SAMPLING: " + Arrays.toString(network.calculate(sample_input)));
        System.out.println("SAMPLING: " + Arrays.toString(sample_correct_output));

    }

    public static TrainSet createTrainSet(int start, int end) {


        int breaker = 0;      //remove TODO
        TrainSet set = new TrainSet(28 * 28, 10);

        try {

            String path = new File("").getAbsolutePath();

            MnistImageFile m = new MnistImageFile(path + "/resource/trainImage.idx3-ubyte", "rw");
            MnistLabelFile l = new MnistLabelFile(path + "/resource/trainLabel.idx1-ubyte", "rw");

            for (int i = start; i <= end; i++) {
                if (i % 100 == 0) {
                    System.out.println("prepared: " + i);
                }

                double[] input = new double[28 * 28];
                double[] output = new double[10];

                output[l.readLabel()] = 1d;
                for (int j = 0; j < 28 * 28; j++) {
                    input[j] = (double) m.read() / (double) 256;
                }


                if (breaker == 0) {
                    sample_input = input.clone();
                    sample_correct_output=output.clone();
                    breaker++;
                }
                set.addData(input, output);
                m.next();
                l.next();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return set;
    }

    public static void trainData(Network net, TrainSet set, int epochs, int loops, int batch_size) {
        for (int e = 0; e < epochs; e++) {
            net.train(set, loops, batch_size);
            System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>   " + e + "   <<<<<<<<<<<<<<<<<<<<<<<<<<");
        }
    }

    public static void testTrainSet(Network net, TrainSet set, int printSteps) {
        int correct = 0;
        for (int i = 0; i < set.size(); i++) {
            double highest = indexOfHighestValue(net.calculate(set.getInput(i)));
            double actualHighest = indexOfHighestValue(set.getOutput(i));
            if (highest == actualHighest) {

                correct++;
            }
            if (i % printSteps == 0) {
                System.out.println(i + ": " + (double) correct / (double) (i + 1));
            }
        }
        System.out.println("Testing finished, RESULT: " + correct + " / " + set.size() + "  -> " + (double) correct / (double) set.size() + " %");
    }

    public static int indexOfHighestValue(double[] values) {
        int index = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[index]) {
                index = i;
            }
        }
        return index;
    }
}
