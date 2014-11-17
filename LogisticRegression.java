package machine_learning;

import java.util.List;
import java.util.Random;

public class LogisticRegression {
	public static int MAX_ITER_TIMES = 500;
	public static double BASE_ALPHA = 0.01;
	public int N;                 // dimension.
	public int M;                 // number of samples.
	public double[] theta;        //parameters.
	
	/**
	 * @Description:  This class assumes x0 = 1.0.
	 * @param data_set
	 * @param isBGD
	 */
	public LogisticRegression(List<DataEntry<double[]>> data_set, boolean isBGD) {
		this.M = data_set.size();
		this.N = data_set.get(0).data_vec.length;
		this.theta = new double[N];
		if(isBGD)
			batchGradientDescend(data_set);
		else
			stochasticGradientDescend(data_set);
	}
	
	/**
	 * 
	 * @Title:        batchGradientDescend 
	 * @Description:  TODO 
	 * @param:        @param data_set    
	 * @return:       void    
	 * @throws 
	 * @author        UncleLee
	 */
	private void batchGradientDescend(List<DataEntry<double[]>> data_set){
		double[] errorCache = new double[M];
		for(int iter=0;iter<MAX_ITER_TIMES;iter++){
			for(int i=0;i<M;i++){
				double[] vec = data_set.get(i).data_vec;
				int label = data_set.get(i).label;
				errorCache[i] = GeneralFunction.sigmoid(MatrixUtils.vectorDotMultiply(theta, vec)) - label;
			}
			for(int j=0;j<N;j++){
				double error = 0.;
				for(int i=0;i<M;i++)
					error += errorCache[i]*data_set.get(i).data_vec[j];
				theta[j] -= error*BASE_ALPHA;
			}
		}
	}
	
	/**
	 * 
	 * @Title:        stochasticGradientDescend 
	 * @Description:  TODO 
	 * @param:        @param data_set    
	 * @return:       void    
	 * @throws 
	 * @author        UncleLee
	 */
	private void stochasticGradientDescend(List<DataEntry<double[]>> data_set){
		Random random = new Random();
		for(int iter=0;iter<MAX_ITER_TIMES;iter++){
			for(int i=0;i<M;i++){
				int index = random.nextInt(M);
				double[] vec = data_set.get(index).data_vec;
				int label = data_set.get(index).label;
				double alpha =  4.0/(1.0+iter+i)+BASE_ALPHA;
				double errorTemp = GeneralFunction.sigmoid((MatrixUtils.vectorDotMultiply(vec, theta))) - label;
				for(int j=0;j<N;j++)
					theta[j] -= errorTemp*vec[j]*alpha;
			}
		}
	}
}
