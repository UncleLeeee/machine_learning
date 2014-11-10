package machine_learning;

import java.util.Random;

public class KDTreeTest {
	
	public static double LOW_BOND = -1000.;
	public static double HIGH_BOND = 1000.;
	
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
	
	
	public static void main(String[] args) {
		int M = 100;
		int N = 10;
		int testNums = 100;
		double[][] data = KDTreeTest.getData(M, N);
		double errorRate = 1e-8;
		KDTree tree = new KDTree(data);
		for(int i=0;i<testNums;i++){
			double[] test = getSample(N);
			double val1 = getNearest(data, test);
			tree.searchNearest(test);
			double val2 = tree.nearestDistance;
			if(Math.abs(val1-val2)>errorRate)
				System.out.printf("%d th sample is wrong, the val1's value is %f, the val2's value is %f \n",i,val1,val2);
		}
	}
}
