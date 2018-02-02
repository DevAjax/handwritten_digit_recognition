import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Network {
    public double[][] output;  //[layer_number][neuron_number]
    public double[][][] weight; //[layer_number][neuron_number][previous_neuron_number]
    public double[][] bias; //[layer_number][neuron_number]


    public final int[] NETWORK_LAYERS_SIZES; //etc. [3,4,1,5]: input layer - size 3,first hidden layer - size 4, second hidden layer - size 1, output layer - size 5
    public final int INPUT_LAYER_SIZE; //numbers of the neurons in the input layer
    public final int OUTPUT_LAYER_SIZE; //numbers of the neurons in the output layer
    public final int NETWORK_SIZE;  //sum of the hidden layers plus 2 (input, output)


    public Network(int... NETWORK_LAYERS_SIZES) {
        this.NETWORK_LAYERS_SIZES = NETWORK_LAYERS_SIZES;
        this.INPUT_LAYER_SIZE = NETWORK_LAYERS_SIZES[0];
        this.OUTPUT_LAYER_SIZE = NETWORK_LAYERS_SIZES[NETWORK_LAYERS_SIZES.length - 1];
        this.NETWORK_SIZE = NETWORK_LAYERS_SIZES.length;


        this.output = new double[NETWORK_SIZE][];
        this.weight = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = random1dArray(NETWORK_LAYERS_SIZES[i]);
            this.bias[i] = random1dArray(NETWORK_LAYERS_SIZES[i]);

            if (i > 0) {
                this.weight[i] = random2dArray(NETWORK_LAYERS_SIZES[i], NETWORK_LAYERS_SIZES[i - 1]);
            } else {
                this.weight[0] = null;
            }
        }


    }

    public double[] calculation(double... input) {
        if (input.length != INPUT_LAYER_SIZE) {
            return null;
        }
        this.output[0] = input; //initialize output data
        for (int layers = 1; layers < NETWORK_SIZE; layers++) {
            for (int neuron = 0; neuron < NETWORK_LAYERS_SIZES[layers]; neuron++) {

                double arg = bias[layers][neuron];

                for (int previousNeuron = 0; previousNeuron < NETWORK_LAYERS_SIZES[layers - 1]; previousNeuron++) {
                    arg += output[layers - 1][previousNeuron] * weight[layers][neuron][previousNeuron];
                }

                output[layers][neuron] = sigmoid(arg);
            }
        }
        return output[NETWORK_SIZE - 1];
    }

    public double sigmoid(double d) {
        return 1d / (1 + Math.exp(-d));
    }


    public static void main(String args[]) {
        Network net = new Network(4, 1, 3, 4);
        double[] output = net.calculation(0.2, 0.1, 0.3, 0.5);
        System.out.println(Arrays.toString(output));
        System.out.println(Arrays.toString(net.bias));
        System.out.println(Arrays.toString(net.output));


    }


    public double[] random1dArray(int size) {
        double arr[] = new double[size];
        for (int i = 0; i < size; i++) {
            arr[i] = Math.random() * (0.1 - 0.9);
        }
        return arr;
    }

    public double[][] random2dArray(int x, int y) {
        double arr[][] = new double[x][y];
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                arr[i][j] = 1;
            }
        }

        return arr;
    }

}
