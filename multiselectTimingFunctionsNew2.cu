// -*- c++ -*-

/* Based on timingFunctions.cu */
#include <stdlib.h>


#ifndef GPUNUMBER
  #define GPUNUMBER 0
#endif

#define MAX_THREADS_PER_BLOCK 1024

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {    \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)



template <typename T>
 struct results_t {
  float time;
  T * vals;
};

template <typename T>
void setupForTiming(cudaEvent_t &start, cudaEvent_t &stop, T * h_vec, T ** d_vec, results_t<T> ** result, uint numElements, uint kCount) {
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMalloc(d_vec, numElements * sizeof(T));
  cudaMemcpy(*d_vec, h_vec, numElements * sizeof(T), cudaMemcpyHostToDevice);

  *result = (results_t<T> *) malloc (sizeof (results_t<T>));
  (*result)->vals = (T *) malloc (kCount * sizeof (T));
}

template <typename T>
void wrapupForTiming(cudaEvent_t &start, cudaEvent_t &stop, float time, results_t<T> * result) {
  result->time = time;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  //   cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////
//          THE SORT AND CHOOSE TIMING FUNCTION
/////////////////////////////////////////////////////////////////


template <typename T>
__global__ void copyInChunk(T * outputVector, T * inputVector, uint * kList, uint kListCount, uint numElements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < kListCount) 
    outputVector[idx] = inputVector[numElements - kList[idx]];
  
}

template <typename T>
inline void bestSort(T * d_vec, const uint numElements) {
  cubDeviceSort<T>(d_vec, numElements);
}


template <>
inline void bestSort<double>(double * d_vec, const uint numElements) {
  mgpuDeviceSort<double>(d_vec, numElements);
}



template<typename T>
results_t<T>* timeSortAndChooseMultiselect(T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

  cudaEventRecord(start, 0);
  bestSort<T>(d_vec, numElements);
  
  T * d_output;
  uint * d_kList;

  cudaMalloc (&d_output, kCount * sizeof (T));
  cudaMalloc (&d_kList, kCount * sizeof(uint));
  cudaMemcpy (d_kList, kVals, kCount * sizeof (uint), cudaMemcpyHostToDevice);

  int threads = MAX_THREADS_PER_BLOCK;
  if (kCount < threads)
    threads = kCount;
  int blocks = (int) ceil (kCount / (float) threads);

  copyInChunk<T><<<blocks, threads>>>(d_output, d_vec, d_kList, kCount, numElements);
  cudaMemcpy (result->vals, d_output, kCount * sizeof (T), cudaMemcpyDeviceToHost);

  //printf("first result: %u \n", result->vals);

  cudaFree(d_output);
  cudaFree(d_kList); 
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}




/////////////////////////////////////////////////////////////////
//          BUCKETMULTISELECT TIMING FUNCTION
/////////////////////////////////////////////////////////////////


template<typename T>
results_t<T>* timeBucketMultiselect (T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, GPUNUMBER);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
 
  cudaEventRecord(start, 0);

  // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
  BucketMultiselect::bucketMultiselectWrapper(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}



template<typename T>
results_t<T>* timeBucketMultiselectNew2 (T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, GPUNUMBER);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
 
  cudaEventRecord(start, 0);

  // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
  BucketMultiselectNew2::bucketMultiselectWrapper(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}


/////////////////////////////////////////////////////////////////
//          cpu QUICKMULTISELECT TIMING FUNCTION
/////////////////////////////////////////////////////////////////



// FUNCTION TO TIME CPU Based QUICKMULTISELECT
template<typename T>
results_t<T>* timeQuickMultiselect (T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  T * h_vec_copy;
  h_vec_copy = (T *) malloc(sizeof(T)*numElements);
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, GPUNUMBER);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
  cudaMemcpy(h_vec_copy, h_vec, numElements*sizeof(T), cudaMemcpyHostToHost);
 
  cudaEventRecord(start, 0);

  quickMultiselectWrapper(h_vec_copy, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}




/////////////////////////////////////////////////////////////////
//          Library Specific SORT AND CHOOSE TIMING FUNCTIONS
/////////////////////////////////////////////////////////////////

// FUNCTION TO TIME SORT&CHOOSE WITH CUB RADIX SORT
template<typename T>
results_t<T>* timeCUBSortAndChooseMultiselect(T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

  cudaEventRecord(start, 0);
  
  cubDeviceSort(d_vec, numElements);

  T * d_output;
  uint * d_kList;

  cudaMalloc (&d_output, kCount * sizeof (T));
  cudaMalloc (&d_kList, kCount * sizeof(uint));
  cudaMemcpy (d_kList, kVals, kCount * sizeof (uint), cudaMemcpyHostToDevice);

  int threads = MAX_THREADS_PER_BLOCK;
  if (kCount < threads)
    threads = kCount;
  int blocks = (int) ceil (kCount / (float) threads);

  copyInChunk<T><<<blocks, threads>>>(d_output, d_vec, d_kList, kCount, numElements);
  cudaMemcpy (result->vals, d_output, kCount * sizeof (T), cudaMemcpyDeviceToHost);

  cudaFree(d_output);
  cudaFree(d_kList); 

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}




//FUNCTION TO TIME ModernGPU Sort and Choose 
template<typename T>
results_t<T>* timeMGPUSortAndChooseMultiselect(T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

  cudaEventRecord(start, 0);
  mgpuDeviceSort(d_vec, numElements);
  
  T * d_output;
  uint * d_kList;

  cudaMalloc (&d_output, kCount * sizeof (T));
  cudaMalloc (&d_kList, kCount * sizeof(uint));
  cudaMemcpy (d_kList, kVals, kCount * sizeof (uint), cudaMemcpyHostToDevice);

  int threads = MAX_THREADS_PER_BLOCK;
  if (kCount < threads)
    threads = kCount;
  int blocks = (int) ceil (kCount / (float) threads);

  copyInChunk<T><<<blocks, threads>>>(d_output, d_vec, d_kList, kCount, numElements);
  cudaMemcpy (result->vals, d_output, kCount * sizeof (T), cudaMemcpyDeviceToHost);

  cudaFree(d_output);
  cudaFree(d_kList); 
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}





/////////////////////////////////////////////////////////////////
//          Library Specific BUCKETMULTISELECT TIMING FUNCTIONS
/////////////////////////////////////////////////////////////////

// FUNCTION TO TIME CUB BUCKET MULTISELECT
template<typename T>
results_t<T>* timeBucketMultiselect_cub (T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, GPUNUMBER);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
 
  cudaEventRecord(start, 0);

  // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
  BucketMultiselect_cub::bucketMultiselectWrapper_cub(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}


// FUNCTION TO TIME MGPU BUCKET MULTISELECT
template<typename T>
results_t<T>* timeBucketMultiselect_mgpu (T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, GPUNUMBER);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
 
  cudaEventRecord(start, 0);

  // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
  BucketMultiselect_mgpu::bucketMultiselectWrapper_mgpu(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}


// FUNCTION TO TIME THRUST BUCKET MULTISELECT
// This is the original function; it does not use binary search trees.
template<typename T>
results_t<T>* timeBucketMultiselect_thrust (T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, GPUNUMBER);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
 
  cudaEventRecord(start, 0);

  // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
  BucketMultiselect_thrust::bucketMultiselectWrapper_thrust(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}


// FUNCTION TO TIME NAIVE BUCKET MULTISELECT (Does not use kernel density estimator nor binary search trees; not recommended)
template<typename T>
results_t<T>* timeNaiveBucketMultiselect (T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

  cudaEventRecord(start, 0);
  thrust::device_ptr<T> dev_ptr(d_vec);
  thrust::sort(dev_ptr, dev_ptr + numElements);

  for (int i = 0; i < kCount; i++)
    cudaMemcpy(result->vals + i, d_vec + (numElements - kVals[i]), sizeof (T), cudaMemcpyDeviceToHost);
   
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}


/***************************************
********* TOP K SELECT ALGORITHMS
****************************************/

template<typename T>
results_t<T>* timeSortAndChooseTopkselect(T * h_vec, uint numElements, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

  cudaEventRecord(start, 0);
  bestSort<T>(d_vec, numElements);

  cudaMemcpy(result->vals, d_vec, kCount * sizeof(T), cudaMemcpyDeviceToHost);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}

// FUNCTION TO TIME RANDOMIZED TOP K SELECT
template<typename T>
results_t<T>* timeRandomizedTopkselect (T * h_vec, uint numElements, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, GPUNUMBER);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
 
  cudaEventRecord(start, 0);
  result->vals = randomizedTopkSelectWrapper(d_vec, numElements, kCount);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}

// FUNCTION TO TIME BUCKET TOP K SELECT
template<typename T>
results_t<T>* timeBucketTopkselect (T * h_vec, uint numElements, uint kCount) {
  // initialize ks
  uint * kVals = (uint *) malloc(kCount*sizeof(T));
  for (uint i = 0; i < kCount; i++)
    kVals[i] = i+1;

  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, GPUNUMBER);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
 
  cudaEventRecord(start, 0);

  BucketMultiselect::bucketMultiselectWrapper(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}
