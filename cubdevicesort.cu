// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

#include <stdio.h>
#include <iostream>
#include <algorithm>

#include <cub-1.3.2/cub/block/block_load.cuh>
#include <cub-1.3.2/cub/block/block_store.cuh>
#include <cub-1.3.2/cub/block/block_radix_sort.cuh>
#include <cub-1.3.2/cub/device/device_radix_sort.cuh>

using namespace cub;


// A wrapper for the cub specific preprocessing to sort a vector and return the vector ascending sorted order
template <typename T>
void cubDeviceSort (T* d_vec, uint numElements)
{
  //  Allocate memory for the working vector in the cub::DoubleBuffer
  //  Values are already in d_vec.  Assuming default initialization of d_keys.selector = 0 
  T *d_vec_alt;
  CubDebugExit(cudaMalloc(&d_vec_alt, sizeof(T) * numElements));
  cub::DoubleBuffer<T> d_keys(d_vec, d_vec_alt);


  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, numElements));
  // Allocate temporary storage
  CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  // Run sorting operation
  CubDebugExit(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, numElements));
  
  // If the radix sort ended with the sorted vector in the alternate buffer, we need to fix that.
  if (d_keys.selector == 1){
    CubDebugExit(cudaMemcpy(d_vec, d_keys.d_buffers[d_keys.selector], sizeof(T) * numElements, cudaMemcpyDeviceToDevice));
  }


  cudaFree(d_temp_storage); 
  cudaFree(d_vec_alt); 

  return;
}


