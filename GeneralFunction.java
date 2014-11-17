package machine_learning;

public class GeneralFunction {
	public static double sigmoid(double in){
		return 1./(1+Math.exp(-in));
	}
}
