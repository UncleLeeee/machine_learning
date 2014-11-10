package machine_learning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Stack;

public class KDTree {
	// The inner node of kdtree.
	class KDTreeNode{
		public KDTreeNode left = null;
		public KDTreeNode right = null;
		public double[] vec = null;
		public int split_dimension;
		
		public int mark = -1;                //0 indicates left, 1 indicates right.
		
		public KDTreeNode(double[] v, int split) {
			this.vec = v;
			this.split_dimension = split;
		}
		
		@Override
		public String toString() {
			int len = vec.length;
			StringBuilder sb = new StringBuilder();
			sb.append("(");
			for(int i=0;i<len;i++){
				sb.append(vec[i]);
				if(i != len-1)
					sb.append(",");
			}
			sb.append(")");
			return sb.toString();
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
	private KDTreeNode rootNode = null;
	
	/**
	 * 
	 * @param data
	 * @param split
	 * @return
	 */
	private KDTreeNode buildTree(List<Pair> data, final int split){
		if(data == null || data.size() == 0)
			return null;
		Comparator<Pair> comparator = new Comparator<Pair>(){
			public int compare(Pair o1, Pair o2) {
				double v1 = o1.data_vector[split];
				double v2 = o2.data_vector[split];
				return v1<=v2?-1:1;
			}
		};
		Collections.sort(data, comparator);
		int middle = data.size()/2;
		KDTreeNode root = new KDTreeNode(data.get(middle).data_vector, split);
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
	 */
	public KDTree(double[][] data) {
		initTree(data);
	}
	/**
	 * 
	 * @param data
	 */
	private void initTree(double[][] data){
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
	
	public double nearestDistance;
	public double[] nearestData;
	
	/**
	 * 
	 * @Title:        searchNearestRecursively 
	 * @Description:  TODO 
	 * @param:        @param vec
	 * @param:        @param root    
	 * @return:       void    
	 * @throws 
	 */
	private void searchNearestRecursively(double[] vec, KDTreeNode root){
		if(root == null)
			return;
		int split = root.split_dimension;
		double dist = MatrixUtils.calcDist(vec, root.vec);
		if(dist<nearestDistance){
			nearestDistance = dist;
			nearestData = root.vec;
		}
		if(vec[split]<=root.vec[split])
			searchNearestRecursively(vec, root.left);
		else
			searchNearestRecursively(vec, root.right);
		if(Math.abs(vec[split]-root.vec[split])<nearestDistance){
			if(vec[split]<=root.vec[split])
				searchNearestRecursively(vec, root.right);
			else
				searchNearestRecursively(vec, root.left);
		}
	}
	
	/**
	 * 
	 * @Title:        searchNearest 
	 * @Description:  TODO 
	 * @param:        @param vec    
	 * @return:       void    
	 * @throws 
	 */
	public void searchNearest(double[] vec){
		nearestDistance = Double.MAX_VALUE;
		nearestData = null;
		searchNearestRecursively(vec, rootNode);
	}
	/**
	 * 
	 * @Title:        getRootNode 
	 * @Description:  TODO 
	 * @param:        @return    
	 * @return:       KDTreeNode    
	 * @throws 
	 * @author        UncleLee
	 */
	public KDTreeNode getRootNode() {
		return rootNode;
	}
	
	public static void main(String[] args) {
		double[][] data = {{2., 0.}, {6.5,4.},{9.,6.},{4.,117.},{8.,5},{7.,2.}};
		KDTree kd = new KDTree(data);
		KDTreeNode root = kd.getRootNode();
		double[] vec = {3,23};
		kd.searchNearest(vec);
		double res = kd.nearestDistance;
		double[] d = kd.nearestData;
		System.out.println(res);
		for(int i=0;i<d.length;i++)
			System.out.print(" "+d[i]);
	}
}
