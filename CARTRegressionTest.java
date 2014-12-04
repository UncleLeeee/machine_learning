package machine_learning;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class CARTRegressionTest {
	public static String file_name = "C:\\Users\\UncleLee\\Desktop\\myProject\\MLiA_SourceCode\\Ch09\\exp2.txt";
	
	public static void main(String[] args) throws FileNotFoundException {
		Scanner scanner = new Scanner(new File(file_name));
		List<DataEntry<double[], Double>> data_set = new ArrayList<DataEntry<double[],Double>>();
		while(scanner.hasNext()){
			double[] data = new double[2];
			data[0] = 1.0;
			data[1] = scanner.nextDouble();
			double label = scanner.nextDouble();
			DataEntry<double[], Double> entry = new DataEntry<double[], Double>(data, label);
			data_set.add(entry);
		}
		CARTRegression cart = new CARTRegression();
		cart.train(data_set);
		double[] test_set = {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0};
		for(int i=0;i<test_set.length;i++){
			double[] data = new double[2];
			data[0] = 1.0;
			data[1] = test_set[i];
			System.out.println(cart.test(data));
		}
	}
}
