package machine_learning;

public class DataEntry<T> {
	public T data_vec;
	public int label;
	
	public DataEntry(T data, int l) {
		this.data_vec = data;
		this.label = l;
	}
}
