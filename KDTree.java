package machine_learning;

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
	// The number of data entries.
	private int M;
	// DataSet.
	private double[][] dataSet;
	
	public void buildTree(double[][] data){
	}
	
	public void initTree(double[][] data){
		this.M = data.length;
		this.N = data[0].length;
		this.dataSet = data;
	}
}
