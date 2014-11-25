package machine_learning;

import java.util.List;

public class AdaBoost {

}

class DecisionStump{
	public double alpha; 		// weight.
	public int dimension;       // the chosen dimension.
	public double threshold;	// decision bound.
	public boolean is_less;     // the decision direction.
	
	public double error;		// the error of this classifier.
	public int[] predict_val;   // the label predicted by this classifier.
	
	public static final int STEPS = 10;
	
	public static int[] classify(List<DataEntry<double[]>> data_set, int d, double val, boolean is_less){
		int M = data_set.size();
		int[] ret_labels = new int[M];
		for(int i=0;i<M;i++){
			double curr_val = data_set.get(i).data_vec[d];
			if(is_less)
				ret_labels[i] = curr_val<val?-1:1;
			else
				ret_labels[i] = curr_val<val?1:-1;
		}
		return ret_labels;
	}

	public static DecisionStump buildStump(List<DataEntry<double[]>> data_set, double[] weight){
		int M = data_set.size();
		int N = data_set.get(0).data_vec.length;
		int[] labels = new int[M];
		for(int i=0;i<M;i++){
			labels[i] = data_set.get(i).label;
		}
		DecisionStump ret_stump = new DecisionStump();
		double min_error = Double.MAX_VALUE;
		for(int d=0;d<N;d++){
			double max = Double.MIN_VALUE;
			double min = Double.MAX_VALUE;
			for(int i=0;i<M;i++){
				double curr_val = data_set.get(i).data_vec[d];
				max = Math.max(curr_val, max);
				min = Math.min(curr_val, min);
			}
			double step_size = (max-min)/(double)STEPS;
			double curr_val = min;
			for(int i=0;i<step_size;i++,curr_val+=step_size){
				boolean is_less = true;
				for(int ii=0;ii<2;ii++,is_less = !is_less){
					int[] pred_labels = classify(data_set, d, curr_val, is_less);
					double pred_errors = 0.0;
					for(int m=0;m<M;m++){
						if(pred_labels[m]!=labels[m])
							pred_errors += weight[m];
					}
				}
			}
		}
		
		return ret_stump;
	}
	
}
