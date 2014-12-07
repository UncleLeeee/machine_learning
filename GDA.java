package machine_learning;

import java.util.List;

import org.ejml.simple.SimpleMatrix;

public class GDA {
	public int N;
	public int K;
	public double[] phi;
	public double[][] mean;
	public SimpleMatrix var_transpose;
	
	public GDA(int N, int K) {
		this.N = N;
		this.K = K;
		this.phi = new double[K];
		this.mean = new double[K][N];
		this.var_transpose = new SimpleMatrix(new double[N][N]);
	}
	
	public void train(List<DataEntry<double[], Integer>> data_set){
		int m = data_set.size();
		for(DataEntry<double[], Integer> entry:data_set){
			phi[entry.label] += 1;
			for(int i=0;i<N;i++){
				mean[entry.label][i] += entry.data_vec[i];
			}
		}
		for(int i=0;i<K;i++){
			for(int j=0;j<N;j++){
				mean[i][j] /= phi[i];
			}
			phi[i] /= m;
		}
		for(DataEntry<double[], Integer> entry:data_set){
			double[][] data = new double[1][N];
			for(int i=0;i<N;i++)
				data[0][i] = entry.data_vec[i] - this.mean[entry.label][i];
			SimpleMatrix curr_matrix = new SimpleMatrix(data);
			var_transpose.plus(curr_matrix.transpose().mult(curr_matrix));
		}
		var_transpose.scale(1./(double)(m));
		this.var_transpose = var_transpose.invert();
	}
	
	public int test(double[] data){
		double max = Double.MIN_VALUE;
		int ret = -1;
		SimpleMatrix temp;
		for(int k=0;k<K;k++){
			double[][] curr_data = new double[1][N];
			for(int i=0;i<N;i++)
				curr_data[0][i] = data[i] - this.mean[k][i];
			SimpleMatrix curr_matrix = new SimpleMatrix(curr_data);
			temp = curr_matrix.mult(var_transpose).mult(curr_matrix.transpose());
			double r = temp.get(0, 0);
			double res = Math.exp(-0.5*r)*phi[k];
			if(res>max){
				max = res;
				ret = k;
			}
		}
		return ret;
	}
}
