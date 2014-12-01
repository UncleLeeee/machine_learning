package machine_learning;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class LogisticRegressionTest {
	public static String file_name = "C:\\Users\\UncleLee\\Desktop\\myProject\\MLiA_SourceCode\\Ch05\\testSet.txt";

	public static void main(String[] args) throws FileNotFoundException {
		Scanner scanner = new Scanner(new File(file_name));
		List<DataEntry<double[], Double>> data_set = new ArrayList<DataEntry<double[], Double>>();
		while(scanner.hasNext()){
			double[] data = new double[3];
			data[0] = 1.0;
			data[1] = scanner.nextDouble();
			data[2] = scanner.nextDouble();
			DataEntry<double[], Double> entry = new DataEntry<double[], Double>(data, (double)scanner.nextInt());
			data_set.add(entry);
		}
		
		LogisticRegression r = new LogisticRegression(data_set,false);
		int right = 0;
		for(int i=0;i<data_set.size();i++){
			double[] vec = data_set.get(i).data_vec;
			int label = (int)(double)data_set.get(i).label;
			int nLabel = r.test(vec);
			if(label == nLabel)
				right++;
		}
		System.out.printf("%d / %d",right,data_set.size());
	}

}
