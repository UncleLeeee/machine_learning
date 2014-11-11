package machine_learning;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class DecisionTreeID3 {
	
	class DecisionNode{
		public boolean isLeaf = false;
		public int label;
		public Map<Integer, DecisionNode> childs = new HashMap<Integer, DecisionNode>();
	}
	
	public static int calcMajority(int[] data){
		int max = 0;
		Map<Integer, Integer> counter = new HashMap<Integer, Integer>();
		for(int i=0;i<data.length;i++){
			if(counter.containsKey(data[i]))
				counter.put(data[i], counter.get(data[i])+1);
			else
				counter.put(data[i], 1);
		}
		for(Entry<Integer, Integer> entry:counter.entrySet()){
			int val = entry.getValue();
			if(val>max)
				max = val;
		}
		return max;
	}
	
	public static double calcEntropy(int[] data){
		double entropy = 0.;
		int all = data.length;
		Map<Integer, Integer> counter = new HashMap<Integer, Integer>();
		for(Integer i:data){
			if(counter.containsKey(data[i]))
				counter.put(data[i], counter.get(data[i])+1);
			else
				counter.put(data[i], 1);
		}
		for(Entry<Integer, Integer> entry:counter.entrySet()){
			double prob = ((double)entry.getValue())/(double)all;
			entropy += -prob*Math.log(prob);
		}
		return entropy;
	}
	
	public DecisionNode buildTree(List<int[]> data_set, int[] label){
		
	}
}
