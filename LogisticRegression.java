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
}
