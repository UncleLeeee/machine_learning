package machine_learning;

import java.util.List;

public class MatrixUtils {
	public static double calcDist(double[] a1, double[] a2){
		double ret = 0;
		int len = a1.length;
		for(int i=0;i<len;i++)
			ret += (a1[i]-a2[i])*(a1[i]-a2[i]);
		return Math.sqrt(ret);
	}
	
	/**
	 * 
	 * @Title:        matrixMinAndMax 
	 * @Description:  TODO 
	 * @param:        @param dataSet
	 * @param:        @param index
	 * @param:        @param axis
	 * @param:        @return    
	 * @return:       double[]    
	 * @throws 
	 * @author        UncleLee
	 */
	public static double[] matrixMinAndMax(List<double[]> dataSet, int index, int axis){
		double[] ret = new double[2];
		ret[0] = Double.MAX_VALUE;
		ret[1] = Double.MIN_VALUE;
		int m = dataSet.size();
		int n = dataSet.get(0).length;
		if(axis == 0){
			for(int i=0;i<m;i++){
				double val = dataSet.get(i)[index];
				if(val<ret[0])
					ret[0] = val;
				if(val>ret[1])
					ret[1] = val;
			}
		}else{
			for(int i=0;i<n;i++){
				double val = dataSet.get(index)[i];
				if(val<ret[0])
					ret[0] = val;
				if(val>ret[1])
					ret[1] = val;
			}
		}
		return ret;
	}
	
	/**
	 * 
	 * @Title:        matrixMeanAndVariance 
	 * @Description:  TODO 
	 * @param:        @param dataSet
	 * @param:        @param index
	 * @param:        @param axis
	 * @param:        @return    
	 * @return:       double[]    
	 * @throws 
	 * @author        UncleLee
	 */
	public static double[] matrixMeanAndVariance(List<double[]> dataSet, int index, int axis){
		double[] ret = new double[2];
		ret[0] = 0.0;
		ret[1] = 0.0;
		int m = dataSet.size();
		int n = dataSet.get(0).length;
		if(axis == 0){
			for(int i=0;i<m;i++){
				double val = dataSet.get(i)[index];
				ret[0] += val;
				ret[1] += val*val;
			}
			ret[0] /= m;
			ret[1] /= m;
			ret[1] -= ret[0]*ret[0];
		}else{
			for(int i=0;i<n;i++){
				double val = dataSet.get(index)[i];
				ret[0] += val;
				ret[1] += val*val;
			}
			ret[0] /= n;
			ret[1] /= n;
			ret[1] -= ret[0]*ret[0];
		}
		return ret;
	}
	
	public static double vectorDotMultiply(double[] a1, double[] a2){
		double res = 0.;
		for(int i=0;i<a1.length;i++){
			res += a1[i]*a2[i];
		}
		return res;
	}
}
