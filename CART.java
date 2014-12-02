package machine_learning;

import java.util.ArrayList;
import java.util.List;

public class CART<T> {
	class SplitPair{
		public int idx;
		public double val;
		
		public SplitPair(int i, int v) {
			this.idx = i;
			this.val = v;
		}
	}
	
	public List<DataEntry<double[], T>>[] binSplitData(List<DataEntry<double[], T>> data_set, SplitPair sp){
		List<DataEntry<double[], T>>[] ret = new ArrayList[2];
		ret[0] = new ArrayList<DataEntry<double[],T>>();
		ret[1] = new ArrayList<DataEntry<double[],T>>();
		int m = data_set.size();
		for(int i=0;i<m;i++){
			DataEntry<double[], T> entry = data_set.get(i);
			if(entry.data_vec[sp.idx]<sp.val)
				ret[0].add(entry);
			else
				ret[1].add(entry);
		}
		return ret;
	}
	
	public SplitPair chooseBestFeature(List<DataEntry<double[], T>> data_set){
		int m = data_set.size();
	}
	
	
}

class CARTNode<T>{
	public boolean is_leaf = false;					// indicates if this node is a leaf. 
	public int split_index;							// the index of the split feature.
	public double split_val;						// the thredshold of the split feature. 
	public LinearRegressionModel<T> model = null;	// the liear model of this node.
	
	public CARTNode<T> left = null;
	public CARTNode<T> right = null;
}
