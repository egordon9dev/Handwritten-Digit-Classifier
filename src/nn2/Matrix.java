package nn2;

public class Matrix {
    public static double[][] mult(double m1[][], double m2[][]) throws ArrayIndexOutOfBoundsException{
        double output[][] = new double[m1.length][m2[0].length];
        for(int y1 = 0; y1 < m1.length; y1++) {
            for(int x2 = 0; x2 < m2[0].length; x2++) {
                double sum = 0;
                for(int x1 = 0; x1 < m1[0].length; x1++) {
                    sum += m1[y1][x1] * m2[x1][x2];
                }
                output[y1][x2] = sum;
            }
        }
        return output;
    }
    public static double[][] add(double m1[][], double m2[][]) {
         double output[][] = new double[m1.length][m1[0].length];
         
         if(m1.length != m2.length) System.out.println("d2 idx mismatch " + m1.length + " " + m2.length);
         for(int i = 0; i < m1.length; i++) {
             if(m1[i].length != m2[i].length) System.out.println("d1 idx mismatch " + m1[i].length + " " + m2[i].length);
             for(int j = 0; j < m1[0].length; j++) {
                 output[i][j] = m1[i][j] + m2[i][j];
             }
         }
         return output;
    }
    public static double[][][] add(double m1[][][], double m2[][][]) {
         if(m1.length != m2.length) System.out.println("d3 idx mismatch " + m1.length + " " + m2.length);
         double output[][][] = new double[m1.length][][];
         for(int i = 0; i < m1.length; i++) {
             output[i] = Matrix.add(m1[i],m2[i]);
         }
         return output;
    }
    public static double[][] subtract(double m1[][], double m2[][]) {
         double output[][] = new double[m1.length][m1[0].length];
         for(int i = 0; i < m1.length; i++) {
             for(int j = 0; j < m1[0].length; j++) {
                 output[i][j] = m1[i][j] - m2[i][j];
             }
         }
         return output;
    }
    public static double[][] scalarMult(double scalar, double m[][]) {
         double output[][] = new double[m.length][m[0].length];
         for(int i = 0; i < m.length; i++) {
             for(int j = 0; j < m[0].length; j++) {
                 output[i][j] = scalar * m[i][j];
             }
         }
         return output;
    }
    public static double[][] entrywiseProduct(double m1[][], double m2[][]) {
        double output[][] = new double[m1.length][m1[0].length];
         for(int i = 0; i < m1.length; i++) {
             for(int j = 0; j < m1[0].length; j++) {
                 output[i][j] = m1[i][j] * m2[i][j];
             }
         }
         return output;
    }
    public static double[][] transpose(double m[][]) {
        double output[][] = new double[m[0].length][m.length];
        for(int i = 0; i < m.length; i++) {
            for(int j = 0; j < m[0].length; j++) {
                output[j][i] = m[i][j];
            }
        }
        return output;
    }
    public static double[][] fillWith(double m[][], double x) {
        for(int i = 0; i < m.length; i++) {
            for(int j = 0; j < m[0].length; j++) {
                m[i][j] = x;;
            }
        }
        return m;
    }
    public static void print2d(double arr[][]) {
        System.out.println("{");
        for (int i = 0; i < arr.length; i++) {
            System.out.print("  { ");
            for(int j = 0; j < arr[0].length; j++) {
                System.out.print(arr[i][j]);
                if(j < arr[0].length - 1) System.out.print(", ");
            }
            System.out.println("}");
        }
        System.out.println("}");
    }
    public static void print3d(double arr[][][]) {
        System.out.println("{");
        for (int i = 0; i < arr.length; i++) {
            System.out.print("  { ");
            for(int j = 0; j < arr[i].length; j++) {
                System.out.print("{");
                for(int k = 0; k < arr[i][j].length; k++) {
                    System.out.printf("%.2g", arr[i][j][k]);
                    if(k < arr[i][j].length - 1) System.out.print(",");
                }
                System.out.print("}");
                if(j < arr[i].length - 1) System.out.print(", ");
            }
            System.out.println("}");
        }
        System.out.println("}");
    }
}
