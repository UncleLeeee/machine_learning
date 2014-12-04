package machine_learning;

import java.util.List;

public class LinearRegression extends LinearRegressionModel<Double> {

	public LinearRegression(List<DataEntry<double[], Double>> data_set,
			boolean isBGD) {
		super(data_set, isBGD, new LinearGeneral());
	}

	@Override
	public Double test(double[] data) {
		return this.calculator.calc(data, theta);
	}

	@Override
	public Double[] batchTest(List<DataEntry<double[], Double>> data_set) {
		int m = data_set.size();
		Double[] labels = new Double[m];
		for(int i=0;i<m;i++){
			labels[i] = this.calculator.calc(data_set.get(i).data_vec, theta);
		}
		return labels;
	}

	@Override
	protected void calcError(List<DataEntry<double[], Double>> data_set) {
		double mean = 0.0;
		int m = data_set.size();
		for(int i=0;i<m;i++){
			mean += data_set.get(i).label;
		}
		mean /= m;
		double pred_var = 0.0;
		double orig_var = 0.0;
		Double[] pred_labels = batchTest(data_set);
		for(int i=0;i<m;i++){
			pred_var += Math.pow((pred_labels[i]-data_set.get(i).label), 2);
			orig_var += Math.pow((data_set.get(i).label-mean), 2);
		}
		this.fitness = (1.0 - pred_var/orig_var);
		this.error_sum = pred_var;
	}

}
