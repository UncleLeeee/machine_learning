package machine_learning;

public class LinearSigmoid implements LinearCalculator{

	@Override
	public double calc(double[] data, double[] theta) {
		return GeneralFunction.sigmoid(MatrixUtils.vectorDotMultiply(data, theta));
	}

}
