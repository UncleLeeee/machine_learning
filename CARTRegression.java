package machine_learning;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class CARTRegression{
	class SplitPair{
		public int idx;
		public double val;
		
		public SplitPair(int i, int v) {
			this.idx = i;
			this.val = v;
		}
	}
	
	public static int DEFAULT_MIN_NUMS = 10;			// if the size of dataset is less than this value, then stop split.
	public static double MIN_TOLERANCE = 0.98;			// if the accuracy is larger than this value, then stop split.
	public static double MIN_INCREMENT_RATE = 0.80;
	
	private CARTNode<Double> root;
	
	/**
	 * 
	 * @Title:        binSplitData 
	 * @Description:  TODO 
	 * @param:        @param data_set
	 * @param:        @param sp
	 * @param:        @return    
	 * @return:       List<DataEntry<double[],Double>>[]    
	 * @throws 
	 * @author        UncleLee
	 */
	private List<DataEntry<double[], Double>>[] binSplitData(List<DataEntry<double[], Double>> data_set, SplitPair sp){
		List<DataEntry<double[], Double>>[] ret = new ArrayList[2];
		ret[0] = new ArrayList<DataEntry<double[],Double>>();
		ret[1] = new ArrayList<DataEntry<double[],Double>>();
		int m = data_set.size();
		for(int i=0;i<m;i++){
			DataEntry<double[], Double> entry = data_set.get(i);
			if(entry.data_vec[sp.idx]<sp.val)
				ret[0].add(entry);
			else
				ret[1].add(entry);
		}
		return ret;
	}
	
	/**
	 * 
	 * @Title:        chooseBestFeature 
	 * @Description:  TODO 
	 * @param:        @param data_set
	 * @param:        @return    
	 * @return:       SplitPair    
	 * @throws 
	 * @author        UncleLee
	 */
	private SplitPair chooseBestFeature(List<DataEntry<double[], Double>> data_set, double error_sum){
		int m = data_set.size();
		if(m <= DEFAULT_MIN_NUMS)
			return null;
		int n = data_set.get(0).data_vec.length;
		SplitPair best_sp = null;
		SplitPair sp = new SplitPair(-1, -1);
		double min_error = Double.MAX_VALUE;
		for(int i=0;i<n;i++){
			sp.idx = i;
			Set<Double> val_set = new HashSet<Double>();
			for(int j=0;j<m;j++)
				val_set.add(data_set.get(j).data_vec[i]);
			for(double val:val_set){
				sp.val = val;
				double error = 0.0;
				List<DataEntry<double[], Double>>[] temp_data = binSplitData(data_set, sp);
				if(temp_data[0].size()>0){
					LinearRegressionModel<Double> left =  new LinearRegression(temp_data[0], false);
					error += left.error_sum;
				}
				if(temp_data[1].size()>0){
					LinearRegressionModel<Double> right =  new LinearRegression(temp_data[1], false);
					error += right.error_sum;
				}
				if(error<min_error){
					best_sp = sp;
					min_error = error;
				}
			}
		}
		if(min_error/error_sum>MIN_INCREMENT_RATE)
			return null;
		return best_sp;
	}
	
	/**
	 * 
	 * @Title:        buildTree 
	 * @Description:  TODO 
	 * @param:        @param data_set
	 * @param:        @return    
	 * @return:       CARTNode<Double>    
	 * @throws 
	 * @author        UncleLee
	 */
	private CARTNode<Double> buildTree(List<DataEntry<double[], Double>> data_set){
		CARTNode<Double> node = new CARTNode<Double>();
		LinearRegressionModel<Double> this_model = new LinearRegression(data_set, false);
		node.model = this_model;
		if(this_model.fitness>=MIN_TOLERANCE){
			return node;
		}
		SplitPair sp = chooseBestFeature(data_set, this_model.error_sum);
		if(sp == null)
			return node;
		List<DataEntry<double[], Double>>[] temp_data = binSplitData(data_set, sp);
		List<DataEntry<double[], Double>> left_data = temp_data[0];
		List<DataEntry<double[], Double>> right_data = temp_data[1];
		if(left_data.size() == 0 || right_data.size() == 0)
			return node;
		node.is_leaf = false;
		node.split_index = sp.idx;
		node.split_val = sp.val;
		node.left = buildTree(left_data);
		node.right = buildTree(right_data);
		return node;
	}
	
	/**
	 * 
	 * @Title:        train 
	 * @Description:  TODO 
	 * @param:        @param data_set    
	 * @return:       void    
	 * @throws 
	 * @author        UncleLee
	 */
	public void train(List<DataEntry<double[], Double>> data_set){
		this.root = buildTree(data_set);
	}
	
	/**
	 * 
	 * @Title:        test 
	 * @Description:  TODO 
	 * @param:        @param data
	 * @param:        @return    
	 * @return:       double    
	 * @throws 
	 * @author        UncleLee
	 */
	public double test(double[] data){
		CARTNode<Double> index = this.root;
		while(!index.is_leaf){
			int i = index.split_index;
			double v = index.split_val;
			if(data[i]<v)
				index = index.left;
			else
				index = index.right;
		}
		return index.model.test(data);
	}
	
}

class CARTNode<T>{
	public boolean is_leaf = true;					// indicates if this node is a leaf. 
	public int split_index;							// the index of the split feature.
	public double split_val;						// the thredshold of the split feature. 
	public LinearRegressionModel<T> model = null;	// the liear model of this node.
	
	public CARTNode<T> left = null;
	public CARTNode<T> right = null;
}
