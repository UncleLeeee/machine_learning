package machine_learning;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class LogisticRegressionTest {

	public static void main(String[] args) throws FileNotFoundException {
		Scanner scanner = new Scanner(new File("C:\\Users\\UncleLee\\Desktop\\myProject\\MLiA_SourceCode\\Ch05\\testSet.txt"));
		List<DataEntry<double[]>> data_set = new ArrayList<DataEntry<double[]>>();
		while(scanner.hasNext()){
			double[] data = new double[3];
			data[0] = 1.0;
			data[1] = scanner.nextDouble();
			data[2] = scanner.nextDouble();
			DataEntry<double[]> entry = new DataEntry<double[]>(data, scanner.nextInt());
			data_set.add(entry);
		}
		
		LogisticRegression r = new LogisticRegression(data_set,true);
		double[] params = r.theta;
		int right = 0;
		for(int i=0;i<data_set.size();i++){
			double[] vec = data_set.get(i).data_vec;
			int label = data_set.get(i).label;
			double res = GeneralFunction.sigmoid(MatrixUtils.vectorDotMultiply(vec, params));
			int nLabel = res<0.5?0:1;
			if(label == nLabel)
				right++;
		}
		System.out.printf("%d / %d",right,data_set.size());
	}

}
