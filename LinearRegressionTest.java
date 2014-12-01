package machine_learning;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class LinearRegressionTest {

	public static String file_name = "C:\\Users\\UncleLee\\Desktop\\myProject\\MLiA_SourceCode\\Ch08\\ex0.txt";

	public static void main(String[] args) throws FileNotFoundException {
		Scanner scanner = new Scanner(new File(file_name));
		List<DataEntry<double[], Double>> data_set = new ArrayList<DataEntry<double[], Double>>();
		while(scanner.hasNext()){
			double[] data = new double[2];
			data[0] = scanner.nextDouble();
			data[1] = scanner.nextDouble();
			DataEntry<double[], Double> entry = new DataEntry<double[], Double>(data, scanner.nextDouble());
			data_set.add(entry);
		}
		LinearRegression r = new LinearRegression(data_set,false);
		System.out.println(r.theta[0]);
		System.out.println(r.theta[1]);
	}

}
