package machine_learning;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

public class DecisionTreeID3 {

	public static int calcMajority(List<DataEntry<Integer>> data_set){
		int max = 0;
		int len = data_set.size();
		Map<Integer, Integer> counter = new HashMap<Integer, Integer>();
		for(int i=0;i<len;i++){
			int label_val = data_set.get(i).label;
			if(counter.containsKey(label_val))
				counter.put(label_val, counter.get(label_val)+1);
			else
				counter.put(label_val, 1);
		}
		for(Entry<Integer, Integer> entry:counter.entrySet()){
			int val = entry.getValue();
			if(val>max)
				max = val;
		}
		return max;
	}
	
	public static double calcEntropy(List<DataEntry<Integer>> data_set){
		double entropy = 0.;
		int all = data_set.size();
		Map<Integer, Integer> counter = new HashMap<Integer, Integer>();
		for(int i=0;i<all;i++){
			int label_val = data_set.get(i).label;
			if(counter.containsKey(label_val))
				counter.put(label_val, counter.get(label_val)+1);
			else
				counter.put(label_val, 1);
		}
		for(Entry<Integer, Integer> entry:counter.entrySet()){
			double prob = ((double)entry.getValue())/(double)all;
			entropy += -prob*Math.log(prob);
		}
		return entropy;
	}
	
	class DecisionNode{
		public boolean is_leaf = false;
		public int label;
		public int split_feature_index;
		public Map<Integer, DecisionNode> childs = new HashMap<Integer, DecisionNode>();
		
		public DecisionNode(int split, boolean leaf) {
			this.split_feature_index = split;
			this.is_leaf = leaf;
		}
	}
	
	private List<DataEntry<Integer>> splitData(List<DataEntry<Integer>> data_set, int index, int value){
		List<DataEntry<Integer>> split_data = new ArrayList<DataEntry<Integer>>();
		for(DataEntry<Integer> data_entry:data_set){
			if(data_entry.data_vec[index] == value)
				split_data.add(data_entry);
		}
		return split_data;
	}
	
	private boolean[] marked;
	private int marked_counter = 0;
	public int M;
	public int N;
	
	private int chooseBestFeature(List<DataEntry<Integer>> data_set){
		int all = data_set.size();
		int best_feature = 0;
		double curr_entropy = calcEntropy(data_set);
		double max_info_gain = Double.MIN_VALUE;
		for(int i=0;i<N;i++){
			if(marked[i])
				continue;
			Set<Integer> unique_value = getUniqueValue(data_set, i);
			double this_entropy = 0.;
			for(Integer val:unique_value){
				List<DataEntry<Integer>> split_data = splitData(data_set, i, val);
				double prob = (double)(split_data.size())/(double)all;
				this_entropy += calcEntropy(split_data)*prob;
			}
			if(curr_entropy-this_entropy>max_info_gain){
				best_feature = i;
				max_info_gain = curr_entropy-this_entropy;
			}
		}
		return best_feature;
	}
	
	private Set<Integer> getUniqueValue(List<DataEntry<Integer>> data_set, int index){
		Set<Integer> set = new HashSet<Integer>();
		for(DataEntry<Integer> data_entry:data_set){
			set.add(data_entry.data_vec[index]);
		}
		return set;
	}
	
	private DecisionNode buildTree(List<DataEntry<Integer>> data_set){
		DecisionNode root = new DecisionNode(-1, false);
		boolean is_uniformed = true;
		for(int i=1;i<data_set.size();i++){
			if(data_set.get(i-1).label != data_set.get(i).label){
				is_uniformed = false;
				break;
			}
		}
		if(is_uniformed){
			root.is_leaf = true;
			root.label = data_set.get(0).label;
			return root;
		}
		if(marked_counter == N-1){
			root.is_leaf = true;
			root.label = calcMajority(data_set);
			return root;
		}
		int best_feature = chooseBestFeature(data_set);
		marked[best_feature] = true;
		marked_counter ++;
		root.split_feature_index = best_feature;
		Set<Integer> unique_value = getUniqueValue(data_set, best_feature);
		for(Integer val:unique_value){
			root.childs.put(val, buildTree(splitData(data_set, best_feature, val)));
		}
		marked[best_feature] = false;
		marked_counter --;
		return root;
	}
	
	private DecisionNode root = null;
	public DecisionTreeID3(List<DataEntry<Integer>> data_set) {
		this.M = data_set.size();
		this.N = data_set.get(0).data_vec.length;
		this.marked = new boolean[N];
		this.root = buildTree(data_set);
	}
	
}
