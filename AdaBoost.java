package machine_learning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class AdaBoost {
	public static double MIN_VALUE = 1e-16;
	public int N;							// dimension of the data.
	public List<DecisionStump> classifiers;	// classifiers.
	public double[] weight;					// the weight of training samples.
	
	/**
	 * 
	 * @param data_set
	 * @param num_of_classifier
	 */
	public void train(List<DataEntry<double[]>> data_set, int num_of_classifier){
		int M = data_set.size();
		this.classifiers = new ArrayList<DecisionStump>();
		this.N = data_set.get(0).data_vec.length;
		this.weight = new double[M];
		double init_val = 1.0/(double)M;
		Arrays.fill(weight, init_val);
		for(int i=0;i<num_of_classifier;i++){
			DecisionStump stump = DecisionStump.buildStump(data_set, weight);
			double error = Math.max(stump.error, MIN_VALUE);
			stump.alpha = 0.5*Math.log((1-error)/error);
			classifiers.add(stump);
			updateWeight(data_set, stump);
		}
	}
	
	public int test(double[] data_vec){
		double output = 0.0;
		for(DecisionStump stump:classifiers){
			int curr_val = stump.classify(data_vec);
			output += stump.alpha*(double)curr_val;
		}
		if(output>0.0)
			return 1;
		else
			return -1;
	}
	
	private void updateWeight(List<DataEntry<double[]>> data_set, DecisionStump curr_stump){
		int M = data_set.size();
		double[] data_cache = new double[M];
		int[] labels = new int[M];
		for(int i=0;i<M;i++)
			labels[i] = data_set.get(i).label;
		int[] pred_labels = curr_stump.predict_val;
		double alpha = curr_stump.alpha;
		double sum = 0.0;
		for(int i=0;i<M;i++){
			double curr;
			if(labels[i] == pred_labels[i])
				curr = this.weight[i]*Math.pow(Math.E, -alpha);
			else
				curr = this.weight[i]*Math.pow(Math.E, alpha);
			data_cache[i] = curr;
			sum += curr;
		}
		for(int i=0;i<M;i++)
			weight[i] = data_cache[i]/sum;
	}
}

class DecisionStump{
	public double alpha; 		// weight.
	public int dimension;       // the chosen dimension.
	public double threshold;	// decision bound.
	public boolean is_less;     // the decision direction.
	
	public double error;		// the error of this classifier.
	public int[] predict_val;   // the label predicted by this classifier.
	
	public int classify(double[] data_vec){
		int ret;
		if(is_less)
			ret = data_vec[dimension]<threshold?-1:1;
		else
			ret = data_vec[dimension]<threshold?1:-1;
		return ret;
	}
	
	public static int STEPS = 10;	// times of iteration.
	
	/**
	 * 
	 * @param data_set
	 * @param d
	 * @param val
	 * @param is_less
	 * @return
	 */
	public static int[] batchClassify(List<DataEntry<double[]>> data_set, int d, double val, boolean is_less){
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
	
	/**
	 * 
	 * @param data_set
	 * @param weight
	 * @return
	 */
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
					int[] pred_labels = batchClassify(data_set, d, curr_val, is_less);
					double pred_errors = 0.0;
					for(int m=0;m<M;m++){
						if(pred_labels[m]!=labels[m])
							pred_errors += weight[m];
					}
					if(pred_errors<min_error){
						min_error = pred_errors;
						ret_stump.dimension = d;
						ret_stump.error = min_error;
						ret_stump.is_less = is_less;
						ret_stump.threshold = curr_val;
						ret_stump.predict_val = pred_labels;
					}
				}
			}
		}
		return ret_stump;
	}
	
}
