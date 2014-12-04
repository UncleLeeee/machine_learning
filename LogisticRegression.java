package machine_learning;

import java.util.List;

public class LogisticRegression extends LinearRegressionModel<Integer> {
	
	/**
	 * @Description:  This class assumes x0 = 1.0.
	 * @param data_set
	 * @param isBGD
	 */
	public LogisticRegression(List<DataEntry<double[], Double>> data_set, boolean isBGD) {
		super(data_set, isBGD, new LinearSigmoid());
	}

	@Override
	public Integer test(double[] data) {
		double res = this.calculator.calc(data, theta);
		return res<0.5?0:1;
	}

	@Override
	public Integer[] batchTest(List<DataEntry<double[], Double>> data_set) {
		int m = data_set.size();
		Integer[] labels = new Integer[m];
		for(int i=0;i<m;i++){
			double res = this.calculator.calc(data_set.get(i).data_vec, theta);
			labels[i] = res<0.5?0:1;
		}
		return labels;
	}

	@Override
	protected void calcError(List<DataEntry<double[], Double>> data_set) {
		int m = data_set.size();
		int right = 0;
		Integer[] pred_labels = batchTest(data_set);
		double pred_var = 0.0;
		for(int i=0;i<m;i++){
			if(pred_labels[i].equals(data_set.get(i).label))
				right ++;
			pred_var += (double)Math.pow(pred_labels[i] - data_set.get(i).label, 2);
		}
		this.fitness = (double)right/(double)m;
		this.error_sum = pred_var;
	}
}
