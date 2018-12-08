package nn2;

import java.io.*;
import java.nio.channels.FileChannel;
import java.nio.MappedByteBuffer;
public class NN2_handwritten_digit_classification {
    Network net;
    short labels[] = new short[50000];
    short pxls[][][] = new short[50000][784][1];
    short testLabels[] = new short[10000];
    short testPxls[][][] = new short[10000][784][1];
    public void setup() {
        int layers[] = { 784, 200, 40, 10 };
        net = new Network(layers);
        try {
            FileInputStream in = new FileInputStream("data/train-labels.idx1-ubyte");
            in.skip(8);
            int nBytes = 0;
            while(nBytes < 50000 && (labels[nBytes] = (short)in.read()) != -1) {
                nBytes++;
            }
            System.out.println("done reading labels");
            nBytes = 0;
            while(nBytes < 10000 && (testLabels[nBytes] = (short)in.read()) != -1) {
                nBytes++;
            }
            System.out.println("done reading test labels");
            in.close();

            final FileChannel channel = new FileInputStream("data/train-images.idx3-ubyte").getChannel();
            MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 16, channel.size()-16);
            short curByte;
            nBytes = 0;
            int imgIdx = 0;
            while((imgIdx = nBytes/784) < 50000) {
                curByte = (short)(buffer.get() & 0xFF);
                pxls[imgIdx][nBytes%784][0] = curByte;
                nBytes++;
            }
            System.out.println("done reading pxls");
            nBytes = 0;
            while((imgIdx = nBytes/784) < 10000) {
                curByte = (short)(buffer.get() & 0xFF);
                testPxls[imgIdx][nBytes%784][0] = curByte;
                nBytes++;
            }
            System.out.println("done reading test pxls");
            in.close();
        } catch(IOException e) {
            System.out.println(e.getMessage());
        }
        System.out.println("done reading data files");
        System.out.print(labels[0]);
        net.train(labels, pxls, testLabels, testPxls, 1000, 100, 0.4);
        /*
        int layers[] = {2, 8, 2};
        net = new Network(layers);
        short trainingData[][][] = { { {0},{0} }, { {0},{1} }, { {1},{0} }, { {1},{1} } };
        short trainingLabels[] = {0,0,0,1};
        short testData[][][] = { { {0},{0} }, { {0},{1} }, { {1},{0} }, { {1},{1} } };
        short testLabels[] = {0,0,0,1};
        net.train(trainingLabels, trainingData, testLabels, testData, 100, 4, 1.0);*/
    }/*
    int j = 0;
    int t0 = -9999;
    String prevKey = "none";
    void keyPressed() {
      if (key == CODED) {
        if (keyCode == RIGHT && !prevKey.equals("RIGHT")) {
          j++;
          prevKey = "RIGHT";
          System.out.println(j + ": " + labels[j]);
        } else if (keyCode == LEFT && !prevKey.equals("LEFT")) {
          j--;
          prevKey = "LEFT";
          System.out.println(j + ": " + labels[j]);
        } 
      }
    }
    void keyReleased() { prevKey = "none"; }*/
    public void draw() {/*
        for(int i = 0; i < pxls[j].length; i++) {
            stroke(pxls[j][i][0]);
            fill(pxls[j][i][0]);
            int col = i % 28;
            int row = i / 28;
            rect(col * 10, row * 10, 10, 10);
        }*/
    }
    public static void main(String args[]) {
        NN2_handwritten_digit_classification nn = new NN2_handwritten_digit_classification();
        nn.setup();
        while(true) {
            nn.draw();
        }
    }
}