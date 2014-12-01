package machine_learning;

public class DataEntry<T1, T2> {
	public T1 data_vec;
	public T2 label;
	
	public DataEntry(T1 data, T2 l) {
		this.data_vec = data;
		this.label = l;
	}
}
