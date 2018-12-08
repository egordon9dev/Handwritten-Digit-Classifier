package nn2;

import java.util.Random;
import org.jblas.*;

public class Network {
    private Random rand = new Random();
    double learningRate = 0.1;
    public DoubleMatrix weights[], biases[];
    int nLayers;
    int layers[];
    //variables for backprop
    private DoubleMatrix activations[], zs[], gradWeightsDataPt[], gradBiasesDataPt[], gradWeightsEst[], gradBiasesEst[];

    public Network(int layers[]) {
        this.layers = layers;
        nLayers = layers.length;
        weights = new DoubleMatrix[nLayers-1];
        for(int i = 0; i < weights.length; i++) {
            weights[i] = DoubleMatrix.randn(layers[i+1], layers[i]);
        }
        biases = new DoubleMatrix[nLayers-1];
        for(int i = 0; i < biases.length; i++) {
            biases[i] = DoubleMatrix.randn(layers[i+1], 1);
        }
        activations = new DoubleMatrix[nLayers];
        zs = new DoubleMatrix[nLayers];
        gradWeightsDataPt = new DoubleMatrix[weights.length];
        gradBiasesDataPt = new DoubleMatrix[biases.length];
        gradWeightsEst = new DoubleMatrix[weights.length];
        gradBiasesEst = new DoubleMatrix[biases.length];
        for(int i = 0; i < weights.length; i++) {
            gradWeightsDataPt[i] = new DoubleMatrix(weights[i].rows, weights[i].columns);
        }
        for(int i = 0; i < biases.length; i++) {
            gradBiasesDataPt[i] = new DoubleMatrix(biases[i].rows, biases[i].columns);
        }
        activations[0] = new DoubleMatrix(layers[0],1);
        zs[0] = new DoubleMatrix(layers[0],1);
    }
    double clamp(double val, double min, double max) {
        if(val > max) return max;
        if(val < min) return min;
        return val;
    }
    public double activate(double input) {
        if(input > 30) return 1;
        if(input < -30) return 0;
        return 1 / (1.0 + Math.pow((float)Math.E, (float)-input));
    }
    public DoubleMatrix activate(DoubleMatrix m) {
        int r = m.rows, c = m.columns;
        DoubleMatrix output = new DoubleMatrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                output.put(i,j, activate(m.get(i,j)));
            }
        }
        return output;
    }
    public double activatePrime(double input) {
        if(Math.abs(input) > 30) return 0;
        double x = Math.pow((float)Math.E, (float)-input);
        return x / Math.pow(1.0 + (float)x, 2);
    }
    public DoubleMatrix activatePrime(DoubleMatrix m) {
        int r = m.rows, c = m.columns;
        DoubleMatrix output = new DoubleMatrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                output.put(i,j, activatePrime(m.get(i,j)));
            }
        }
        return output;
    }
    

    public DoubleMatrix feedForward(DoubleMatrix input) {
        DoubleMatrix output = new DoubleMatrix(input.length,1);
        for(int i = 0; i < input.rows; i++) output.put(i, 0, input.get(i,0));
        for(int i = 0; i < biases.length; i++) {
            try {
                output = activate(weights[i].mmul(output).add(biases[i]));
            } catch(ArrayIndexOutOfBoundsException e) {
                System.out.println(e);
                System.out.println("at feedForward");
                System.exit(0);
            }
        }
        return output;
    }
    //stochastic gradient descent
    public void train(short labels[], short pxls[][][], short testLabels[], short testPxls[][][], int epochs, int miniBatchSize, double learningRate) {
        short miniBatchLabels[] = new short[miniBatchSize];
        short miniBatchPxls[][][] = new short[miniBatchSize][layers[0]][1];
        for(int i = 0; i < epochs; i++) {
            //shuffle
            int remaining = labels.length, idx;
            while(remaining > 0) {
                idx = (int)(Math.random() * remaining--);
                //swap
                short tempLabel = labels[idx];
                short tempImg[][] = pxls[idx];
                labels[idx] = labels[remaining];
                pxls[idx] = pxls[remaining];
                labels[remaining] = tempLabel;
                pxls[remaining] = tempImg;
            }
            int imgIdx = 0;
            while(imgIdx < labels.length) {
                if(imgIdx % (labels.length / 20) == 0) System.out.print(".");
                for(int j = 0; j < miniBatchSize; j++) {
                    miniBatchLabels[j] = labels[imgIdx];
                    miniBatchPxls[j] = pxls[imgIdx];
                    imgIdx++;
                }
                updateMiniBatch(miniBatchLabels, miniBatchPxls, learningRate);
            }
            int numCorrect = 0;
            DoubleMatrix dblPxls = new DoubleMatrix(testPxls[0].length,1);
            for(int w = 0; w < testPxls.length; w++) {
                for(int j = 0; j < testPxls[0].length; j++) {
                    dblPxls.put(j,0, testPxls[w][j][0]);
                }
                DoubleMatrix output = feedForward(dblPxls);
                if(output.argmax() == testLabels[w]) numCorrect++;
            }
            System.out.println("Epoch " + i + " " + numCorrect + " / " + testLabels.length);
        }
    }
    public void updateMiniBatch(short labels[], short pxls[][][], double learningRate) {
        for(int i = 0; i < weights.length; i++) {
            gradWeightsEst[i] = DoubleMatrix.zeros(weights[i].rows,weights[i].columns);
        }
        for(int i = 0; i < biases.length; i++) {
            gradBiasesEst[i] = DoubleMatrix.zeros(biases[i].rows, biases[i].columns);
        }
        for(int i = 0; i < labels.length; i++) {
            backprop(pxls[i], labels[i]);
            for(int j = 0; j < biases.length; j++) {
                gradWeightsEst[j].addi(gradWeightsDataPt[j]);
                gradBiasesEst[j].addi(gradBiasesDataPt[j]);
            }
        }
        double sc = learningRate / (double)labels.length;
        for(int i = 0; i < weights.length; i++) {
            weights[i].subi(gradWeightsEst[i].mmul(sc));
            biases[i].subi(gradBiasesEst[i].mmul(sc));
        }
    }
    private void backprop(short inputs[][], int label) {
        DoubleMatrix targets = new DoubleMatrix(layers[layers.length-1], 1);
        for(int i = 0; i < targets.rows; i++) targets.put(i,0, 0);
        targets.put(label,0, 1);
        
        //feed forward
        for(int i = 0; i < inputs.length; i++) {
            activations[0].put(i,0, inputs[i][0]);
        }
        for(int i = 0; i < biases.length; i++) {
            zs[i+1] = weights[i].mmul(activations[i]).add(biases[i]);
            activations[i+1] = activate(zs[i+1]);
        }
        //back propogation
        DoubleMatrix delta = activations[activations.length-1].sub(targets).mul(activatePrime(zs[zs.length-1]));
        gradBiasesDataPt[biases.length-1] = delta;
        gradWeightsDataPt[weights.length-1] = delta.mmul(activations[activations.length-2].transpose());
        for(int i = 2; i < nLayers; i++) {
            delta = weights[weights.length - i + 1].transpose().mmul(delta).mul(activatePrime(zs[zs.length-i]));
            gradBiasesDataPt[biases.length-i] = delta;
            gradWeightsDataPt[weights.length-i] = delta.mmul(activations[activations.length-i-1].transpose());
        }
    }
}
