// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

#include <stdio.h>
#include <iostream>
#include <algorithm>

#ifndef GPUNUMBER
  #define GPUNUMBER 0
#endif

#include "moderngpu/include/moderngpu.cuh"		// Include all MGPU kernels.
//#include "moderngpu/include/mgpudevice.cuh"
//#include "moderngpu/include/kernels/mergesort.cuh"

using namespace mgpu;


// A wrapper for the cub specific preprocessing to sort a vector and return the vector ascending sorted order
template<typename T>
void mgpuDeviceSort (T* d_vec, uint numElements)  
{
  //int count = (int)numElements;
  ContextPtr context = CreateCudaDevice(GPUNUMBER);

  MergesortKeys(d_vec, numElements, *context); 

  return;
}




