
/*

template<typename T>
struct minmax_pair
{
 T min_val;
 T max_val;
};


template<typename T>
struct MinMax_initializer
{
 __host__ __device__ __forceinline__ 
 minmax_pair<T> operator()(const T &a) const 
  {
    minmax_pair<T> result;
    result.min_val = a;
    result.max_val = a;
    return result;
  }
};


template<typename T>
struct MinMax_operator
{
 __host__ __device__ __forceinline__ 
 minmax_pair<T> operator()(const T &a, const minmax_pair<T> &b) const 
  {
    minmax_pair<T> result;
    result.min_val = cub::MIN(a, b.min_val);
    result.max_val = cub::MAX(a, b.max_val);
    return result;
  }
};
 

template<typename T>
void cubMinMax_alt(T* d_in, T* h_out, const int length)
{
  MinMax_initializer<T> init_op;
  minmax_pair<T> d_out=init_op(d_in[0]);
  //cudaMalloc(&d_out, sizeof(T)*2);
  MinMax_operator<T> minmax_op;

  //initialize d_out
//  cudaMemcpy(d_out, d_in, sizeof(T), cudaMemcpyDeviceToDevice);
//  cudaMemcpy(d_out+1, d_in, sizeof(T), cudaMemcpyDeviceToDevice);
  

  // determine size of memory needed an allocate
  void *d_temp_storage = NULL;
  size_t temp_size = 0;
  cub::DeviceReduce::Reduce(d_temp_storage, temp_size, d_in, &d_out, length, minmax_op);
  cudaMalloc(&d_temp_storage, temp_size);

  // find min and max
  cub::DeviceReduce::Reduce(d_temp_storage, temp_size, d_in, &d_out, length, minmax_op);

  // copy to host 
  cudaMemcpy(h_out, d_out, 2*sizeof(T), cudaMemcpyDeviceToHost);

  // cleanup
  cudaFree(d_temp_storage);
  cudaFree(d_out);
}


*/



template<typename T>
void cubMinMax_alt(T* d_in, T* h_out, const int length)
{
  T* d_out;
  cudaMalloc(&d_out, sizeof(T)*2);

  void *d_temp_storage_min = NULL;
  size_t temp_size_min = 0;
  void *d_temp_storage_max = NULL;
  size_t temp_size_max = 0;

  cudaStream_t *stream = (cudaStream_t *)malloc(2*sizeof(cudaStream_t));
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

  // determine size of memory needed an allocate
  cub::DeviceReduce::Min(d_temp_storage_min, temp_size_min, d_in, d_out, length, stream[0]);
  cudaMalloc(&d_temp_storage_min, 2*temp_size_min);
  cub::DeviceReduce::Min(d_temp_storage_min, temp_size_min, d_in, d_out, length, stream[0]);

  cub::DeviceReduce::Max(d_temp_storage_min+temp_size_min, temp_size_min, d_in, d_out+1, length, stream[1]);

/*
  cub::DeviceReduce::Max(d_temp_storage_max, temp_size_max, d_in, d_out+1, length, stream[1]);
  cudaMalloc(&d_temp_storage_max, temp_size_max);
  cub::DeviceReduce::Max(d_temp_storage_max, temp_size_max, d_in, d_out+1, length, stream[1]);
*/
  // find min 
  //cub::DeviceReduce::Min(d_temp_storage, temp_size, d_in, d_out, length);

  // find max
  // cub::DeviceReduce::Max(d_temp_storage, temp_size, d_in, d_out+1, length);

  // copy to host 
  cudaMemcpy(h_out, d_out, 2*sizeof(T), cudaMemcpyDeviceToHost);

  // cleanup
  cudaFree(d_temp_storage_min);
  cudaFree(d_temp_storage_max);
  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  cudaFree(d_out);
}






