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

}
