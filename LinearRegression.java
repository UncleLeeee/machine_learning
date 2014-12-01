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

}
