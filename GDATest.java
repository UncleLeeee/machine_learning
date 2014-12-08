package machine_learning;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class GDATest {
	public static String file_name = "/home/lee/MLiA_SourceCode/Ch05/testSet.txt";
	
	public static void main(String[] args) throws FileNotFoundException {
		Scanner scanner = new Scanner(new File(file_name));
		List<DataEntry<double[], Integer>> data_set = new ArrayList<DataEntry<double[],Integer>>();
		while(scanner.hasNext()){
			double[] vec = new double[2];
			vec[0] = scanner.nextDouble();
			vec[1] = scanner.nextDouble();
			DataEntry<double[], Integer> entry = new DataEntry<double[], Integer>(vec, scanner.nextInt());
			data_set.add(entry);
		}
		GDA gda = new GDA(2, 2);
		gda.train(data_set);
		int m = data_set.size();
		int right = 0;
		for(DataEntry<double[], Integer> entry:data_set){
			int res = gda.test(entry.data_vec);
			if(res == entry.label)
				right++;
		}
		System.out.printf("%d / %d",right,m);
	}
}
