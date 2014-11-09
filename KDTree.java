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
		public KDTreeNode parent = null;
		public double[] vec = null;
		public int split_dimension;
		
		public KDTreeNode(double[] v, int split) {
			this.vec = v;
			this.split_dimension = split;
		}
		
		public KDTreeNode anotherChild(KDTreeNode n){
			if(n == left)
				return right;
			else if(n == right)
				return left;
			return null;
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
				return v1<v2?-1:v1==v2?0:1;
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
		if(root.left!=null)
			root.left.parent = root;
		root.right = buildTree(right_data, (split+1)%N);
		if(root.right!=null)
			root.right.parent = root;
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
	
	public double nearestDistance;
	public double[] nearestData;
	
	public void searchNearest(double[] vec){
		nearestDistance = Double.MAX_VALUE;
		nearestData = null; 
		Stack<KDTreeNode> paths = new Stack<KDTreeNode>();
		KDTreeNode curr = this.rootNode;
		while(curr!=null){
			paths.push(curr);
			if(vec[curr.split_dimension]<=curr.vec[curr.split_dimension])
				curr = curr.left;
			else
				curr = curr.right;
		}
		while(!paths.isEmpty()){
			KDTreeNode node = paths.pop();
			KDTreeNode parent = node.parent;
			double dis = MatrixUtils.calcDist(node.vec, vec);
			if(dis<nearestDistance){
				nearestDistance = dis;
				nearestData = node.vec;
			}
			if(parent!=null){
				double disToSplit = Math.abs(vec[parent.split_dimension]-parent.vec[parent.split_dimension]);
				if(disToSplit<nearestDistance){
					KDTreeNode another = parent.anotherChild(node);
					if(another != null)
						paths.push(another);
				}
					
			}
		}
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
		double[][] data = {{2., 3.}, {5.,4.},{9.,6.},{4.,7.},{8.,1.},{7.,2.}};
		KDTree kd = new KDTree();
		kd.initTree(data);
		KDTreeNode root = kd.getRootNode();
		double[] vec = {3,4.5};
		kd.searchNearest(vec);
		double res = kd.nearestDistance;
		double[] d = kd.nearestData;
		System.out.println(root);
	}
}
