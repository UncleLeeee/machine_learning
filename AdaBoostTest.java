package machine_learning;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class AdaBoostTest {
	public static String training_file = "C:\\Users\\UncleLee\\Desktop\\myProject\\MLiA_SourceCode\\Ch07\\horseColicTraining2.txt";
	public static String testing_file = "C:\\Users\\UncleLee\\Desktop\\myProject\\MLiA_SourceCode\\Ch07\\horseColicTest2.txt";
	
	public static AdaBoost classifier = new AdaBoost();
	
	public static List<DataEntry<double[], Integer>> readFile(String file_name){
		List<DataEntry<double[], Integer>> data_set = null;
		Scanner scanner;
		try {
			scanner = new Scanner(new File(file_name));
			data_set = new ArrayList<DataEntry<double[], Integer>>();
			while(scanner.hasNext()){
				String[] one_line = scanner.nextLine().split("\t");
				double[] data_vec = new double[one_line.length-1];
				for(int i=0;i<one_line.length-1;i++)
					data_vec[i] = Double.parseDouble(one_line[i]);
				int label = (int)Double.parseDouble(one_line[one_line.length-1]);
				DataEntry<double[], Integer> data_entry = new DataEntry<double[], Integer>(data_vec, label);
				data_set.add(data_entry);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return data_set;
	}
	
	public static void main(String[] args) throws FileNotFoundException {
		List<DataEntry<double[], Integer>> training_set = readFile(training_file);
		classifier.train(training_set, 100);
		List<DataEntry<double[], Integer>> testing_set = readFile(testing_file);
		int right = 0;
		int all = testing_set.size();
		for(DataEntry<double[], Integer> data:testing_set){
			int pred_label = classifier.test(data.data_vec);
			if(pred_label == data.label)
				right++;
		}
		System.out.printf("%d/%d = %f",right,all,(double)right/(double)all);
	}
}
