// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

#include <stdio.h>
#include <iostream>
#include <algorithm>

#include <cub-1.3.2/cub/block/block_load.cuh>
#include <cub-1.3.2/cub/block/block_store.cuh>
#include <cub-1.3.2/cub/block/block_radix_sort.cuh>
#include <cub-1.3.2/cub/device/device_radix_sort.cuh>
#include <cub-1.3.2/cub/util_allocator.cuh>
#include <cub-1.3.2/cub/device/device_scan.cuh>
#include <cub-1.3.2/cub/device/device_select.cuh>
#include <cub-1.3.2/cub/device/device_reduce.cuh>

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


// computes an exclusive sum of d_in (length numElements) and stores it in d_out
// Example: d_in = [1 2 3 4 5], d_out = [0 1 3 6 10], numElements = 5
template <typename T>
void cubDeviceExclusiveSum (T* d_in, T* d_out, uint numElements)
{
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, numElements));
  // Allocate temporary storage
  CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  // Run exclusive sum operation
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, numElements));

  cudaFree(d_temp_storage); 

  return;
}


// computes an inclusive sum of d_in (length numElements) and stores it in d_out
// Example: d_in = [1 2 3 4 5], d_out = [1 3 6 10 15], numElements = 5
template <typename T>
void cubDeviceInclusiveSum (T* d_in, T* d_out, uint numElements)
{
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, numElements));
  // Allocate temporary storage
  CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  // Run inclusive sum operation
  CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, numElements));

  cudaFree(d_temp_storage); 

  return;
}


// computes the sum elements in d_in (length numElements) and stores it in d_out (length 1)
// Example: d_in = [1 2 3 4 5], d_out = [15], numElements = 5
template <typename T>
void cubDeviceReduceSum (T* d_in, T* d_out, uint numElements)
{
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, numElements));
  // Allocate temporary storage
  CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  // Run summation operation
  CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, numElements));

  cudaFree(d_temp_storage); 

  return;
}


// copies the elements in d_in (length numElements) which are marked by a flag in d_flags (length numelements)
// and stores it in d_out and records the length of d_out in d_num_selected_out.
// Example: d_in = [1 2 3 4 5], d_flags = [0 1 0 0 1], d_out = [2 5], numElements = 5, d_num_selected_out = 2
template <typename T>
void cubDeviceSelectFlagged (T* d_in, uint* d_flags, T* d_out, uint *d_num_selected_out, uint numElements)
{
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, numElements));
  // Allocate temporary storage
  CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  // Run select operation
  CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, numElements));

  cudaFree(d_temp_storage); 

  return;
}


// since the selection function must allocate memory, this function combines combines the operation 
// when two lists are going to be selected with the same flags.
// This function is used in findKBucketsByBlock and the output vector are also the input vectors.
void SelectFlagged (uint* d_in_A, uint* d_out_A, uint* d_in_B, uint* d_out_B,  uint* d_flags, uint *d_num_selected_out, uint numElements)
{
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in_A, d_flags, d_out_A, d_num_selected_out, numElements));
  // Allocate temporary storage
  CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  // Run select operation on two lists
  CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in_A, d_flags, d_out_A, d_num_selected_out, numElements));
  CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in_B, d_flags, d_out_B, d_num_selected_out, numElements));

  cudaFree(d_temp_storage); 

  return;
}



