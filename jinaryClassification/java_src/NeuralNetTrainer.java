package java_src;

import java.io.*;
import java.util.*;

public class NeuralNetTrainer {

    public static void main(String[] args) throws IOException {
        double[][] x_train = loadCSV("../data/x_train_filtered.csv");
        double[][] y_train = loadCSV("../data/y_train_filtered.csv");

        int inputUnits = 784, hidden1 = 25, hidden2 = 15, outputUnits = 1;
        double learningRate = 0.01;
        int epochs = 1000;
        int batchSize = 64;

        double[][] W1 = randomInit(inputUnits, hidden1, Math.sqrt(1.0 / inputUnits));
        double[][] W2 = randomInit(hidden1, hidden2, Math.sqrt(1.0 / hidden1));
        double[][] W3 = randomInit(hidden2, outputUnits, Math.sqrt(1.0 / hidden2));

        double[][] b1 = new double[1][hidden1];
        double[][] b2 = new double[1][hidden2];
        double[][] b3 = new double[1][outputUnits];

        long start = System.nanoTime();
        double prevCost = computeCost(y_train, forwardPass(x_train, W1, b1, W2, b2, W3, b3, "relu")[2]);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int i = 0; i < x_train.length; i += batchSize) {
                int endIdx = Math.min(i + batchSize, x_train.length);
                double[][] x_batch = Arrays.copyOfRange(x_train, i, endIdx);
                double[][] y_batch = Arrays.copyOfRange(y_train, i, endIdx);

                double[][][] activations = forwardPass(x_batch, W1, b1, W2, b2, W3, b3, "relu");
                double[][] a1 = activations[0], a2 = activations[1], a3 = activations[2];

                double[][] dz3 = subtract(a3, y_batch);
                double[][] dW3 = scalarMultiply(dot(transpose(a2), dz3), 1.0 / x_batch.length);
                double[][] db3 = meanRows(dz3);

                double[][] dz2 = multiply(dot(dz3, transpose(W3)), reluDerivative(a2));
                double[][] dW2 = scalarMultiply(dot(transpose(a1), dz2), 1.0 / x_batch.length);
                double[][] db2 = meanRows(dz2);

                double[][] dz1 = multiply(dot(dz2, transpose(W2)), reluDerivative(a1));
                double[][] dW1 = scalarMultiply(dot(transpose(x_batch), dz1), 1.0 / x_batch.length);
                double[][] db1 = meanRows(dz1);

                W3 = subtract(W3, scalarMultiply(dW3, learningRate));
                b3 = subtract(b3, scalarMultiply(db3, learningRate));
                W2 = subtract(W2, scalarMultiply(dW2, learningRate));
                b2 = subtract(b2, scalarMultiply(db2, learningRate));
                W1 = subtract(W1, scalarMultiply(dW1, learningRate));
                b1 = subtract(b1, scalarMultiply(db1, learningRate));
            }

            if (epoch % 1 == 0) {
                double cost = computeCost(y_train, forwardPass(x_train, W1, b1, W2, b2, W3, b3, "relu")[2]);
                System.out.printf("Epoch %d: Cost = %.6f\n", epoch, cost);
                saveModel(W1, b1, W2, b2, W3, b3, "../checkpoints/epoch_" + epoch);
            }
        }

        long end = System.nanoTime();
        double seconds = (end - start) / 1e9;
        double finalCost = computeCost(y_train, forwardPass(x_train, W1, b1, W2, b2, W3, b3, "relu")[2]);

        saveModel(W1, b1, W2, b2, W3, b3, "../model");

        System.out.println("\nModel Summary:");
        System.out.printf("Training Time: %.2f seconds\n", seconds);
        System.out.printf("Initial Cost: %.4f\n", prevCost);
        System.out.printf("Final Cost: %.4f\n", finalCost);
        System.out.printf("Cost Reduction: %.2f%%\n", 100 * (prevCost - finalCost) / prevCost);
    }

    public static double[][] subtract(double[][] A, double[][] B) {
        int m = A.length, n = A[0].length;
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[i][j] = A[i][j] - B[i][j];
        return result;
    }

    public static double[][] scalarMultiply(double[][] A, double scalar) {
        int m = A.length, n = A[0].length;
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[i][j] = A[i][j] * scalar;
        return result;
    }

    public static double[][] multiply(double[][] A, double[][] B) {
        int m = A.length, n = A[0].length;
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[i][j] = A[i][j] * B[i][j];
        return result;
    }

    public static double[][] meanRows(double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] mean = new double[1][n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                mean[0][j] += A[i][j];
            }
            mean[0][j] /= m;
        }
        return mean;
    }

    public static double computeCost(double[][] y, double[][] y_hat) {
        int m = y.length;
        double epsilon = 1e-10;
        double cost = 0.0;
        for (int i = 0; i < m; i++) {
            double predicted = Math.min(Math.max(y_hat[i][0], epsilon), 1 - epsilon);
            cost += y[i][0] * Math.log(predicted) + (1 - y[i][0]) * Math.log(1 - predicted);
        }
        return -cost / m;
    }

    public static void saveModel(double[][] W1, double[][] b1,
            double[][] W2, double[][] b2,
            double[][] W3, double[][] b3,
            String folderPath) throws IOException {
        new File(folderPath).mkdirs();
        saveCSV(W1, folderPath + "/W1.csv");
        saveCSV(b1, folderPath + "/b1.csv");
        saveCSV(W2, folderPath + "/W2.csv");
        saveCSV(b2, folderPath + "/b2.csv");
        saveCSV(W3, folderPath + "/W3.csv");
        saveCSV(b3, folderPath + "/b3.csv");
    }

    public static void saveCSV(double[][] data, String filePath) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filePath));
        for (double[] row : data) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < row.length; i++) {
                sb.append(row[i]);
                if (i < row.length - 1)
                    sb.append(",");
            }
            writer.write(sb.toString());
            writer.newLine();
        }
        writer.close();
    }

    public static double[][] loadCSV(String filePath) throws IOException {
        List<double[]> data = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;

        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            double[] row = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                row[i] = Double.parseDouble(parts[i]);
            }
            data.add(row);
        }

        reader.close();

        double[][] result = new double[data.size()][];
        for (int i = 0; i < data.size(); i++) {
            result[i] = data.get(i);
        }

        return result;
    }

    public static double[][] randomInit(int rows, int cols, double scale) {
        Random rand = new Random();
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = rand.nextGaussian() * scale;
            }
        }

        return result;
    }

    public static double[][] dot(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        int p = B[0].length;

        double[][] result = new double[m][p];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    }

    public static double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] result = new double[n][m];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = A[i][j];
            }
        }

        return result;
    }

    public static double[][] sigmoid(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = 1.0 / (1.0 + Math.exp(-Z[i][j]));
            }
        }

        return result;
    }

    public static double[][] relu(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = Math.max(0, Z[i][j]);
            }
        }

        return result;
    }

    public static double[][] sigmoidDerivative(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = A[i][j] * (1 - A[i][j]);
            }
        }

        return result;
    }

    public static double[][] reluDerivative(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = (A[i][j] > 0) ? 1 : 0;
            }
        }

        return result;
    }

    public static double[][] addBias(double[][] Z, double[][] b) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = Z[i][j] + b[0][j];
            }
        }

        return result;
    }

    public static double[][][] forwardPass(double[][] x, double[][] W1, double[][] b1,
            double[][] W2, double[][] b2,
            double[][] W3, double[][] b3,
            String activationHidden) {
        double[][] z1 = addBias(dot(x, W1), b1);
        double[][] a1 = activationHidden.equals("relu") ? relu(z1) : sigmoid(z1);

        double[][] z2 = addBias(dot(a1, W2), b2);
        double[][] a2 = activationHidden.equals("relu") ? relu(z2) : sigmoid(z2);

        double[][] z3 = addBias(dot(a2, W3), b3);
        double[][] a3 = sigmoid(z3);

        return new double[][][] { a1, a2, a3 };
    }
}
