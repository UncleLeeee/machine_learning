package machine_learning;

import java.util.List;

public class KDTree {
	// The inner node of kdtree.
	class TreeNode{
		public TreeNode left = null;
		public TreeNode right = null;
		public double[] vec = null;
		public int split_dimension;
		
		public TreeNode(double[] v, int split) {
			this.vec = v;
			this.split_dimension = split;
		}
	}
	
	// The dimension of data.
	private int N;
	
	public void buildTree(List<double[]> dataSet){
		int M = dataSet.size();
		this.N = dataSet.get(0).length;
		
	}
}
