package matrixmulti;

import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;


public class Main 
{
	static int numberOfThreads = 299;

	public static void main(String[] args) {
      
        double[][] matrix1 = randomSquareMatrix(2000);
        double[][] matrix2 = randomSquareMatrix(2000);
        
        try {
            matrixMultiplicationParallel(matrix1,matrix2);
        } catch (InterruptedException | ExecutionException e) {
        	
            e.printStackTrace();
        }
    }

    public static double[][] randomSquareMatrix(int n){
        double[][] mat = new double[n][n];
        Random rand = new Random();
        for(int i=0; i<n; i++) 
        	for(int j=0; j<n; j++) 
        		mat[i][j]=rand.nextInt(10);
        return mat;
    }

    public static void matrixMultiplicationParallel(double[][] matrix1, double[][] matrix2) 
    		throws InterruptedException, ExecutionException
    {
        int lengthMatrix1 = matrix1.length;
        double[][] resultMatrix = new double[lengthMatrix1][lengthMatrix1];
        
        ExecutorService executorThread = Executors.newFixedThreadPool(numberOfThreads);
        
        Future<Double>[][] result = new Future[lengthMatrix1][lengthMatrix1];
        
        double startTime = System.currentTimeMillis();
       
        for(int i=0; i<lengthMatrix1; i++)
        {
            for(int j=0; j<lengthMatrix1; j++)
            {
                result[i][j] = executorThread.submit(new Multiplication(matrix1[i],matrix2[j]));
            }
        }

        executorThread.shutdown();
        executorThread.awaitTermination(1, TimeUnit.DAYS);

        for(int i=0; i<lengthMatrix1; i++)
        {
            for(int j=0; j<lengthMatrix1; j++)
            {
                resultMatrix[i][j] = result[i][j].get();
            }
        }
        
        System.out.println(((System.currentTimeMillis()-startTime)/1000)+ " seconds with " + numberOfThreads + " threads." );
    }

    public static class Multiplication implements Callable<Double>{

    	Multiplication(double[] vec1, double[] vec2)
        {
            this.vec1=vec1; 
            this.vec2=vec2;
        }
        double result;
        double[] vec1, vec2;

        public Double call() 
        {
            result=0;
            for(int i=0; i<vec1.length; i++) 
            	result +=vec1[i]*vec2[i];
            return result;
        }
    }
}
