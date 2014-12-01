package machine_learning;

public class LinearGeneral implements LinearCalculator {

	@Override
	public double calc(double[] data, double[] theta) {
		return MatrixUtils.vectorDotMultiply(data, theta);
	}

}
