package machine_learning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
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
	
	// The data with an index.
	class Pair{
		public int index;
		public double[] data_vector;
		
		public Pair(int i, int n) {
			index = i;
			data_vector = new double[n];
		}
	}
	
	// The dimension of data.
	private int N;
	// The number of data entries.
	private int M;
	// DataSet.
	private List<Pair> dataSet;
	// RootNode.
	private TreeNode rootNode = null;
	
	/**
	 * 
	 * @param data
	 * @param split
	 * @return
	 */
	private TreeNode buildTree(List<Pair> data, final int split){
		if(data == null || data.size() == 0)
			return null;
		Comparator<Pair> comparator = new Comparator<Pair>(){
			public int compare(Pair o1, Pair o2) {
				double v1 = o1.data_vector[split];
				double v2 = o2.data_vector[split];
				return v1<v2?-1:v1==v2?0:1;
			}
		};
		Collections.sort(data, comparator);
		int middle = data.size()/2;
		TreeNode root = new TreeNode(data.get(middle).data_vector, split);
		List<Pair> left_data = new ArrayList<Pair>();
		List<Pair> right_data = new ArrayList<Pair>();
		for(int i=0;i<middle;i++)
			left_data.add(data.get(i));
		for(int i=middle+1;i<data.size();i++)
			right_data.add(data.get(i));
		root.left = buildTree(left_data, (split+1)%N);
		root.right = buildTree(right_data, (split+1)%N);
		return root;
	}
	
	/**
	 * 
	 * @param data
	 */
	public void initTree(double[][] data){
		this.M = data.length;
		this.N = data[0].length;
		this.dataSet = new ArrayList<Pair>();
		for(int i=0;i<M;i++){
			Pair p = new Pair(i, N);
			dataSet.add(p);
			for(int j=0;j<N;j++)
				 p.data_vector[j] = data[i][j];
		}
		rootNode = buildTree(dataSet, 0);
	}
}
