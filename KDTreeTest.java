package machine_learning;

import java.util.Random;

public class KDTreeTest {
	
	public static double LOW_BOND = -1000.;
	public static double HIGH_BOND = 1000.;
	public static double ERROR_RATE = 1E-8;
	
	static Random rand = new Random();
	
	private static double generateDouble(){
		return (rand.nextDouble()*(HIGH_BOND-LOW_BOND) + LOW_BOND);
	}
	
	public static double[][] getData(int m, int n) {
		double[][] data = new double[m][n];
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				data[i][j] = generateDouble();
			}
		}
		return data;
	}
	
	public static double[] getSample(int n){
		double[] vec = new double[n];
		for(int i=0;i<n;i++)
			vec[i] = generateDouble();
		return vec;
	}
	
	public static double getNearest(double[][] data, double[] vec){
		double ret = Double.MAX_VALUE;
		int m = data.length;
		for(int i=0;i<m;i++){
			double dist = MatrixUtils.calcDist(data[i], vec);
			if(dist<ret)
				ret = dist;
		}
		return ret;
	}
	/**
	 * 
	 * @param m indicates the number of samples.
	 * @param n indicates the dimention of data.
	 * @param testNums indicates the number of test cases.
	 */
	public static void correctnessTest(int m, int n, int testNums){
		double[][] data = getData(m, n);
		KDTree tree = new KDTree(data);
		boolean no_fault = true;
		for(int i=0;i<testNums;i++){
			double[] test = getSample(n);
			double val1 = getNearest(data, test);
			tree.searchNearest(test);
			double val2 = tree.nearestDistance;
			if(Math.abs(val1-val2)>ERROR_RATE){
				System.out.printf("%d th sample is wrong, the val1's value is %f, the val2's value is %f \n",i,val1,val2);
				no_fault = false;
			}
		}
		if(no_fault)
			System.out.println("all samples are correct!");
	}
	/**
	 * 
	 * @param m indicates the number of samples.
	 * @param n indicates the dimention of data.
	 * @param testNums indicates the number of test cases.
	 */
	public static void runningtimeTest(int m, int n, int testNums){
		double[][] data = getData(m, n);
		double[][] testCases = getData(testNums, n);
		KDTree tree = new KDTree(data);
		long start = System.currentTimeMillis();
		for(int i=0;i<testNums;i++){
			getNearest(data, testCases[i]);
		}
		long end = System.currentTimeMillis();
		System.out.println("linear searching running: "+Long.toString(end-start)+"ms");
		start = System.currentTimeMillis();
		for(int i=0;i<testNums;i++){
			tree.searchNearest(testCases[i]);
		}
		end = System.currentTimeMillis();
		System.out.println("kd-tree searching running: "+Long.toString(end-start)+"ms");
	}
	
	
	public static void main(String[] args) {
		int M = 100000;
		int N = 10;
		int testNums = 1000;
//		correctnessTest(M, N, testNums);
		runningtimeTest(M, N, testNums);
	}
}
