// -*- c++ -*-

/* Copyright 2012 Jeffrey Blanchard, Erik Opavsky, and Emircan Uysaler
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <limits>
#include <mpi.h>

#include <math.h>
#include <ctime>
#include "recursionKernels.cu"
#include "findk.cu"


namespace BucketMultiselectNewFindK{
  using namespace std;

#define MAX_THREADS_PER_BLOCK 1024
#define CUTOFF_POINT 200000 
#define TIMING_ON
#define MIN_SLOPE 2 ^ -1022
#define SAFE

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)


  /// ***********************************************************
  /// ***********************************************************
  /// **** HELPER CPU FUNCTIONS
  /// ***********************************************************
  /// ***********************************************************



  /* This timing function uses CUDA event timing to process the amount of time
     required, and print out result with the given index.

     start a timer with option = 0
     stop a timer with option = 1
  */

  
  cudaEvent_t start, stop;
  float time;

  inline void timing_switch(int option, int ind){
    if(option == 0) {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);
    } else {
      cudaThreadSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      printf("Time %d: %lf \n", ind, time);
    }
  }

  inline void timing(int option, int ind){
#ifdef TIMING_ON 
    timing_switch(option, ind);
#endif
  }


  /* This function initializes a vector to all zeros on the host (CPU).
   */
  template<typename T>
  void setToAllZero (T * d_vector, int length) {
    cudaMemset(d_vector, 0, length * sizeof(T));
  }


  void Check_CUDA_Error(const char *message)
  {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "Error: %s: %s\n", message, cudaGetErrorString(error) );
      exit(-1);
    }
  }

  inline void SAFEcuda(const char *message) {
#ifdef SAFE
    Check_CUDA_Error(message);
#endif
  }


  /*
*********************************************************
*/


  /* This function finds the bin containing the kth element we are looking for (works on 
     the host). While doing the scan, it stores the sum-so-far of the number of elements in 
     the ctive buckets containing one of the k order statistics.

     markedBuckets : buckets containing the corresponding k values
     sums : sum-so-far of the number of elements in the buckets where k values fall into
  */
  inline int findKBuckets(uint * d_bucketCount, uint * h_bucketCount, int numBuckets
                          , uint * kVals, int numKs, uint * sums, uint * markedBuckets
                          , int numBlocks) {
    // consider the last row which holds the total counts
    int sumsRowIndex= numBuckets * (numBlocks-1);

    SAFEcuda("pre memcpy");

    CUDA_CALL(cudaMemcpy(h_bucketCount, d_bucketCount + sumsRowIndex, 
                         sizeof(uint) * numBuckets, cudaMemcpyDeviceToHost));
    SAFEcuda("memcpy");

    int kBucket = 0;
    int k;
    int sum = h_bucketCount[0];

    for(register int i = 0; i < numKs; i++) {
      k = kVals[i];
      while ((sum < k) & (kBucket < numBuckets - 1)) {
        kBucket++;
        sum += h_bucketCount[kBucket]; 
      }
      markedBuckets[i] = kBucket;
      sums[i] = sum - h_bucketCount[kBucket];

    }

    return 0;
  }

  // **********************************************************
  // ***********  sort  phase differs by type  ****************
  // ****** mgpu merge sort typically faster for doubles *****
  // **********************************************************
  template <typename T>
  void inline sort_phase (T* Input, const int length) {
    cubDeviceSort<T>(Input, length);
  }


  template <>
  void inline sort_phase<double> (double* Input, const int length) {
    mgpuDeviceSort<double>(Input, length);
  }


  /// ***********************************************************
  /// ***********************************************************
  /// **** HELPER GPU FUNCTIONS-KERNELS
  /// ***********************************************************
  /// ***********************************************************



  /* This function assigns elements to buckets based on the pivots and slopes determined 
     by a randomized sampling of the elements in the vector. At the same time, this 
     function keeps track of count.

     d_elementToBucket : bucket assignment for every array element
     d_bucketCount : number of element that falls into the indexed buckets within the block
  */
  template <typename T>
  __global__ void assignSmartBucket (T * d_vector, int length, int numBuckets
                                     , double * slopes, T * pivots, T * pivottree, int numPivots
                                     , uint* d_elementToBucket , uint* d_bucketCount, int offset) {
  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    uint bucketIndex;
    int threadIndex = threadIdx.x;  

    int numBigBuckets = numPivots - 1;
    
    //variables in shared memory for fast access
    __shared__ int sharedNumSmallBuckets;
    if (threadIndex < 1) 
      sharedNumSmallBuckets = numBuckets / numBigBuckets;

    extern __shared__ uint array[];
    double * sharedSlopes = (double *)array;
    T * sharedPivots = (T *)&sharedSlopes[numPivots];
    T * sharedPivotTree = (T *)&sharedPivots[numPivots];
    uint * sharedBuckets = (uint *)&sharedPivotTree[numPivots];

  
    //reading bucket counts into shared memory where increments will be performed
    for (int i = 0; i < (numBuckets / MAX_THREADS_PER_BLOCK); i++) 
      if (threadIndex < numBuckets) 
        sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex] = 0;

    if(threadIndex < numPivots) {
      *(sharedPivots + threadIndex) = *(pivots + threadIndex);
      *(sharedSlopes + threadIndex) = *(slopes + threadIndex);
      *(sharedPivotTree + threadIndex) = *(pivottree + threadIndex);
    }


    syncthreads();

    //assigning elements to buckets and incrementing the bucket counts
    if(index < length) {
      int i;

      for(i = index; i < length; i += offset) {
        T num = d_vector[i];

        int PivotIndex = 1;

        for(int j=1; j < numBigBuckets; j*=2){
          PivotIndex = (PivotIndex << 1) + (num >= sharedPivotTree[PivotIndex-1]);
        }
        PivotIndex = PivotIndex - numBigBuckets;

	int localBucket = (int) (((double)num - (double)sharedPivots[PivotIndex]) 
                                 * sharedSlopes[PivotIndex]);

        bucketIndex = (PivotIndex * sharedNumSmallBuckets) 
          + localBucket;
        if (bucketIndex == numBuckets) 
          bucketIndex= numBuckets-1;


        d_elementToBucket[i] = bucketIndex;
        atomicInc(sharedBuckets + bucketIndex, length); 

      }
    }

    syncthreads();      

    //reading bucket counts from shared memory back to global memory
    for (int i = 0; i <(numBuckets / MAX_THREADS_PER_BLOCK); i++) {
      if (threadIndex < numBuckets) {
        *(d_bucketCount + blockIdx.x * numBuckets 
          + i * MAX_THREADS_PER_BLOCK + threadIndex) = 
          *(sharedBuckets + i * MAX_THREADS_PER_BLOCK + threadIndex);
      } // end if threadIndex < numBuckets
    } // end for

  } // end function assignSmartBuckets



  /* This function cumulatively sums the count of every block for a given bucket s.t. the
     last block index holds the total number of elements falling into that bucket all over the 
     array.
     updates d_bucketCount
  */
  __global__ void sumCounts(uint * d_bucketCount, const int numBuckets
                            , const int numBlocks) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int j=1; j<numBlocks; j++) 
      d_bucketCount[index + numBuckets*j] += d_bucketCount[index + numBuckets*(j-1)];
  }



  /* This function reindexes the buckets counts for every block according to the 
     accumulated d_reindexCounter counter for the reduced vector.
     updates d_bucketCount
  */
  __global__ void reindexCounts(uint * d_bucketCount, const int numBuckets
                                , const int numBlocks, uint * d_reindexCounter
                                , uint * d_markedBuckets , const int numUniqueBuckets) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIndex<numUniqueBuckets) {
      int index = d_markedBuckets[threadIndex];
      uint add = d_reindexCounter[threadIndex];

      for(int j=0; j<numBlocks; j++) 
        d_bucketCount[index + numBuckets*j] += add;
    }
  }



  /* This function copies the elements of buckets that contain kVals into a newly allocated 
     reduced vector space.
     newArray - reduced size vector containing the essential elements
     *** This function is not used in current implementation, replaced by copyElements_tree. ***
     */
  template <typename T>
  __global__ void copyElements (T* d_vector, int length, uint* elementToBucket
                                , uint * buckets, const int numBuckets, T* newArray, uint offset
                                , uint * d_bucketCount, int numTotalBuckets) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex;
    int loop = numBuckets / MAX_THREADS_PER_BLOCK;

    extern __shared__ uint sharedBuckets[];

    for (int i = 0; i <= loop; i++) {      
      threadIndex = i * blockDim.x + threadIdx.x;
      if(threadIndex < numBuckets) {
        sharedBuckets[threadIndex] = buckets[threadIndex];
      }
    }
    
    syncthreads();

    int minBucketIndex;
    int maxBucketIndex; 
    int midBucketIndex;
    uint temp;
    int compare;

    if(idx < length) {
      for(int i=idx; i<length; i+=offset) {
        temp = elementToBucket[i];
        minBucketIndex = 0;
        maxBucketIndex = numBuckets-1;
        compare = 0;

        //thread divergence avoiding binary search over the markedBuckets to find a match quickly
        for(int j = 1; j < numBuckets; j*=2) {  
          midBucketIndex = (maxBucketIndex + minBucketIndex) / 2;
          compare = (temp > sharedBuckets[midBucketIndex]);
          minBucketIndex = compare ? midBucketIndex : minBucketIndex;
          maxBucketIndex = compare ? maxBucketIndex : midBucketIndex;
        }

        if (buckets[maxBucketIndex] == temp) 
          newArray[atomicDec(d_bucketCount + blockIdx.x * numTotalBuckets 
                             + sharedBuckets[maxBucketIndex], length)-1] = d_vector[i];
      }
    }

  }

  /* This kernel copies the elements of buckets that contain kVals into a newly allocated 
     reduced vector space.
     newArray - reduced size vector containing the essential elements.
     This kernel differs from copyElements in that it loads a binary search tree
     for the unique buckets into shared memory.  It requires more shared memory to properly
     form the tree.  For a small number (< 128) of order statistics, the tree search is not advantageous.
     The main bucketMultiselect can be altered to utilize an if (numUnique < 128) conditional, calling 
     copyElements if true and calling copyElements_tree if false.
  */



  template <typename T>
  __global__ void copyElements_tree (T* d_vector, int length, uint* elementToBucket
                                     , uint * uniqueBuckets, const int numUnique, const int numUnique_extended, T* newArray, uint offset
                                     , uint * d_bucketCount, int numTotalBuckets) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex;
    int loop = (numUnique_extended) / MAX_THREADS_PER_BLOCK;
    int mid = numUnique_extended / 2;
    int blockOffset = blockIdx.x * numTotalBuckets;

    extern __shared__ uint activeTree[];

    int treeidx, level, shift, remainder, bucketidx;
    // read from shared memory into a binary search tree
    for (int i = 0; i <= loop; i++) {      
      threadIndex = i * blockDim.x + threadIdx.x;
      if (threadIndex < numUnique_extended) {
        treeidx = threadIndex+1;
        level = (int) floorf ( log2f( (float)treeidx ) );
        shift = (1 << level);
        remainder = treeidx - shift;

        bucketidx = ((2*remainder + 1)*mid) / shift;
        if (bucketidx < numUnique) {
          activeTree[threadIndex] = uniqueBuckets[bucketidx];
        } else {
          activeTree[threadIndex] = uniqueBuckets[numUnique-1];
        } // end if (bucketidx) {} else
      } // end if (threadIndex)
    }  // end for
    
    syncthreads();

    int temp_bucket, temp_active, treeindex, active, searchdepth;

    // binary search tree through the active buckets to see
    // if the current element is in an active bucket.  
    // If not, active = 0. If so, active = 1.
    if(idx < length) {
      for(int i=idx; i<length; i+=offset) {
        temp_bucket = elementToBucket[i];
        treeindex = 1;
        active = 0;
        searchdepth = 1;
        while ( (active==0) && (searchdepth<numUnique_extended) ){
          temp_active = activeTree[treeindex - 1];
          searchdepth *= 2;
          (temp_active == temp_bucket) ? active++ : ( treeindex = (treeindex << 1) + (temp_bucket > temp_active) );
        }  // endwhile


        // if this element is in an active bucket, copy it to the new input vector
        if (active) {
          newArray[atomicDec(d_bucketCount + blockOffset + temp_active, length)-1] = d_vector[i];
          //printf ("copied: %f\n", d_vector[i]);
        }  // end if (active)
      }  // ends for loop with offset jump
    } // ends if (idx < length)
  }  // ends copyElements_tree kernel



  template <typename T>
  __global__ void checkBuckets (int numBuckets, double* slopes, int* d_bucketCount
                                , int numBlocks, T* orderStats, T* d_vector, uint* markedBuckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // One thread per bucket

    if (idx < numBuckets) {

      // Check whether the max - min is less than double-precision tolerance
	
      if (d_bucketCount[numBuckets * (numBlocks - 1) + threadIdx.x] / slopes[idx] < MIN_SLOPE) {

	// Sum the last row of d_bucketCount to find the number of elements in the previous
	// buckets
	
	int cumulativeCount = 0;
	for (int i = 0; i < blockIdx.x+1; i++) {
          cumulativeCount += d_bucketCount[numBuckets * (numBlocks - 1) + i];
	}
        orderStats[idx] = d_vector[cumulativeCount];
        markedBuckets[idx] = 0;
      }
    }
  }


  /* This function speeds up the copying process the requested kVals by clustering them
     together.
  */
  template <typename T>
  __global__ void copyValuesInChunk (T * outputVector, T * inputVector, uint * kList
                                     , uint * kIndices, int kListCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int loop = kListCount / MAX_THREADS_PER_BLOCK;

    for (int i = 0; i <= loop; i++) {      
      if (idx < kListCount) {
        *(outputVector + *(kIndices + idx)) = *(inputVector + *(kList + idx) - 1);
        printf ("copied from %d\n", *(kList + idx) - 1);
      }
    }
  }


  /* This kernel sums the d_totalBucketCount vector
   */
  template <typename T>
  __global__ void sumTotalCounts (uint* d_totalBucketCount, int numBlocks, int numBuckets
                                  , int world_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint length = numBlocks * numBuckets;
    int offset = blockDim.x * numBlocks;
    int sum;
    for (int i = idx; i < length; i += offset) {
      sum = 0;
      for (int j = 0; j < world_size; j++) {
        int k = d_totalBucketCount[idx + (numBlocks * numBuckets * j)];
        //sum += d_totalBucketCount[idx + (numBlocks * numBuckets * j)];
        sum += k;
        //printf ("adding %d from %d  ", k, idx + numBlocks * numBuckets * j);
      }
      //printf ("added %d at %d    ", sum, idx);
      *(d_totalBucketCount + idx) = sum;
    }
  }


  /// ***********************************************************
  /// ***********************************************************
  /// **** GENERATE PIVOTS
  /// ***********************************************************
  /// ***********************************************************



  /* Hash function using Monte Carlo method
   */
  __host__ __device__
  unsigned int hash(unsigned int a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
  }



  /* RandomNumberFunctor
   */
  struct RandomNumberFunctor :
    public thrust::unary_function<unsigned int, float> {
    unsigned int mainSeed;

    RandomNumberFunctor(unsigned int _mainSeed) :
    mainSeed(_mainSeed) {}
  
    __host__ __device__
    float operator()(unsigned int threadIdx)
    {
      unsigned int seed = hash(threadIdx) * mainSeed;

      thrust::default_random_engine rng(seed);
      rng.discard(threadIdx);
      thrust::uniform_real_distribution<float> u(0, 1);

      return u(rng);
    }
  };



  /* This function creates a random vector of 1024 elements in the range [0 1]
   */
  template <typename T>
  void createRandomVector(T * d_vec, int size) {
    timeval t1;
    uint seed;

    gettimeofday(&t1, NULL);
    seed = t1.tv_usec * t1.tv_sec;
  
    thrust::device_ptr<T> d_ptr(d_vec);
    thrust::transform (thrust::counting_iterator<uint>(0), 
                       thrust::counting_iterator<uint>(size), 
                       d_ptr, RandomNumberFunctor(seed));
  }



  /* This function maps the [0 1] range to the [0 vectorSize] and 
     grabs the corresponding elements.
  */
  template <typename T>
  __global__ void enlargeIndexAndGetElements (T * in, T * list, int size) {
    *(in + blockIdx.x*blockDim.x + threadIdx.x) = 
      *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
  }

  __global__ void enlargeIndexAndGetElements (float * in, uint * out, uint * list, int size) {
    *(out + blockIdx.x * blockDim.x + threadIdx.x) = 
      (uint) *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
  }



  /* This function generates Pivots from the random sampled data and calculates slopes.
 
     pivots - arrays of pivots
     slopes - array of slopes
  */
  template <typename T>
  void generatePivots (uint * pivots, uint * pivottree, double * slopes, uint * d_list, int sizeOfVector
                       , int numPivots, int sizeOfSample, int totalSmallBuckets, uint min, uint max) {
  
    float * d_randomFloats;
    uint * d_randomInts;
    int endOffset = 22;
    int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
    int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

    cudaMalloc (&d_randomFloats, sizeof (float) * sizeOfSample);
  
    d_randomInts = (uint *) d_randomFloats;

    createRandomVector (d_randomFloats, sizeOfSample);

    // converts randoms floats into elements from necessary indices
    enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK)
      , MAX_THREADS_PER_BLOCK>>>(d_randomFloats, d_randomInts, d_list, 
                                 sizeOfVector);

    pivots[0] = min;
    pivots[numPivots-1] = max;

    cubDeviceSort<T>(d_randomInts, sizeOfSample);

    cudaThreadSynchronize();

    // set the pivots which are next to the min and max pivots using the random element 
    // endOffset away from the ends
    cudaMemcpy (pivots + 1, d_randomInts + endOffset - 1, sizeof (uint)
                , cudaMemcpyDeviceToHost);
    cudaMemcpy (pivots + numPivots - 2, d_randomInts + sizeOfSample - endOffset - 1, 
                sizeof (uint), cudaMemcpyDeviceToHost);
    slopes[0] = numSmallBuckets / (double) (pivots[1] - pivots[0]);

    for (register int i = 2; i < numPivots - 2; i++) {
      cudaMemcpy (pivots + i, d_randomInts + pivotOffset * (i - 1) + endOffset - 1, 
                  sizeof (uint), cudaMemcpyDeviceToHost);
      slopes[i - 1] = numSmallBuckets / (double) (pivots[i] - pivots[i - 1]);
    }

    slopes[numPivots - 3] = numSmallBuckets / 
      (double) (pivots[numPivots - 2] - pivots[numPivots - 3]);
    slopes[numPivots - 2] = numSmallBuckets / 
      (double) (pivots[numPivots - 1] - pivots[numPivots - 2]);

    int level = numPivots - 1;
    int shift = 0;
    for (register int j=1; j < (numPivots - 1); j*=2){
      level >>= 1;
      for (register int k=0; k <= shift; k++){
        pivottree[shift + k] = pivots[(2*k+1)*level];
      }
      shift = (shift << 1) | 1;
    }

    cudaFree(d_randomFloats);
  }  // end generatePivots
  
  template <typename T>
  void generatePivots (T * pivots, T * pivottree, double * slopes, T * d_list, int sizeOfVector
                       , int numPivots, int sizeOfSample, int totalSmallBuckets, T min, T max) {
    T * d_randoms;
    int endOffset = 22;
    int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
    int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

    cudaMalloc (&d_randoms, sizeof (T) * sizeOfSample);
  
    createRandomVector (d_randoms, sizeOfSample);

    // converts randoms floats into elements from necessary indices
    enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK)
      , MAX_THREADS_PER_BLOCK>>>(d_randoms, d_list, sizeOfVector);

    pivots[0] = min;
    pivots[numPivots - 1] = max;

    cubDeviceSort<T>(d_randoms, sizeOfSample);

    cudaThreadSynchronize();

    // set the pivots which are endOffset away from the min and max pivots
    cudaMemcpy (pivots + 1, d_randoms + endOffset - 1, sizeof (T), 
                cudaMemcpyDeviceToHost);
    cudaMemcpy (pivots + numPivots - 2, d_randoms + sizeOfSample - endOffset - 1, 
                sizeof (T), cudaMemcpyDeviceToHost);
    slopes[0] = numSmallBuckets / ((double)pivots[1] - (double)pivots[0]);
    
    for (register int i = 2; i < numPivots - 2; i++) {
      cudaMemcpy (pivots + i, d_randoms + pivotOffset * (i - 1) + endOffset - 1, 
                  sizeof (T), cudaMemcpyDeviceToHost);
      slopes[i - 1] = numSmallBuckets / ((double) pivots[i] - (double) pivots[i - 1]);
    }

    slopes[numPivots - 3] = numSmallBuckets / 
      ((double)pivots[numPivots - 2] - (double)pivots[numPivots - 3]);
    slopes[numPivots - 2] = numSmallBuckets / 
      ((double)pivots[numPivots - 1] - (double)pivots[numPivots - 2]);

    // **** extra space in slopes
    slopes[numPivots - 1]=0;

    int level = numPivots - 1;
    int shift = 0;
    for (register int j=1; j < (numPivots - 1); j*=2){
      level >>= 1;
      for (register int k=0; k <= shift; k++){
        pivottree[shift + k] = pivots[(2*k+1)*level];
      }
      shift = (shift << 1) | 1;
    }

    cudaFree(d_randoms);
  } // end generatePivots<uint>



  /// ***********************************************************
  /// ***********************************************************
  /// **** bucketMultiSelect: the main algorithm
  /// ***********************************************************
  /// ***********************************************************

  /* This function is the main process of the algorithm. It reduces the given multi-selection
     problem to a smaller problem by using bucketing ideas.
  */
  template <typename T>
  T bucketMultiSelect (T* d_vector, int length, uint * kVals, int numKs, T * output, int blocks
                       , int threads, int numBuckets, int numPivots, int world_rank, int world_size
                       , char* processor_name, uint datatype) {    

    //CUDA_CALL(cudaDeviceReset());

    /// ***********************************************************
    /// **** STEP 1: Initialization 
    /// **** STEP 1.1: Find Min and Max of the whole vector
    /// **** We don't need to go through the rest of the algorithm if it's flat
    /// ***********************************************************
    timing(0,1);


    // for (int i = 0; i < numKs; i++)
    //   printf ("kVals[%d] = %u\n", i, kVals[i]);

    // SHOULD THIS HAPPEN DISTRIBUTED?

    //find max and min with thrust
   
    T maximum, minimum;

    thrust::device_ptr<T>dev_ptr(d_vector);
    thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result = 
      thrust::minmax_element(dev_ptr, dev_ptr + length);

    minimum = *result.first;
    maximum = *result.second;

    //if the max and the min are the same, then we are done
    // if (maximum == minimum) {
    //   for (register int i = 0; i < numKs; i++) 
    //     output[i] = minimum;
      
    //   return 1;
    // }
    

    /// ***********************************************************
    /// **** STEP 1: Initialization 
    /// **** STEP 1.2: Declare variables and allocate memory
    /// **** Declare Variables
    /// ***********************************************************

    //declaring variables for kernel launches
    int threadsPerBlock = threads;
    int numBlocks = blocks;
    int offset = blocks * threads;

    // variables for the randomized selection
    int sampleSize = 1024;

    // pivot variables
    double slopes[numPivots - 1];
    double * d_slopes;
    T pivots[numPivots];
    T * d_pivots;
    T pivottree[numPivots];
    T * d_pivottree;

    //Allocate memory to store bucket assignments
    size_t size = length * sizeof(uint);
    uint * d_elementToBucket;    //array showing what bucket every element is in

    CUDA_CALL(cudaMalloc(&d_elementToBucket, size));

    //Allocate memory to store bucket counts

    size_t totalBucketSize = numBlocks * numBuckets * sizeof(uint);
    uint totalBucketLength = numBlocks * numBuckets;
    uint * h_bucketCount = (uint *) malloc (numBuckets * sizeof (uint));
    uint * h_localBucketCount = (uint*) malloc (numBuckets * sizeof(uint));
    uint* h_trueBucketCount = (uint*) malloc (totalBucketSize);

    //array showing the number of elements in each bucket
    uint * d_bucketCount; 

    CUDA_CALL(cudaMalloc(&d_bucketCount, totalBucketSize));

    // array showing the number of elements in each bucket. Will remain local to a given processor.
    uint* d_localBucketCount;

    CUDA_CALL(cudaMalloc(&d_localBucketCount, totalBucketSize));

    uint* h_totalBucketCount;
    uint* d_totalBucketCount;

    if (world_rank == 0) {
      h_totalBucketCount = (uint*) malloc (totalBucketSize * world_size);
      CUDA_CALL (cudaMalloc(&d_totalBucketCount, totalBucketSize * world_size));
    }
    else
      h_totalBucketCount = (uint*) malloc (totalBucketSize);
      
    MPI_Barrier(MPI_COMM_WORLD);

    // array of kth buckets
    int numUniqueBuckets;
    uint * d_kVals; 
    uint kthBuckets[numKs]; 
    uint kthBucketScanner[numKs]; 
    uint * kIndices = (uint *) malloc (numKs * sizeof (uint));
    uint * d_kIndices;
    uint uniqueBuckets[numKs];
    uint * d_uniqueBuckets; 
    uint reindexCounter[numKs];  
    uint *localReindexCounter = (uint*) malloc (numKs * sizeof(uint));
    uint * d_reindexCounter;
    uint * d_localReindexCounter;
    //    int precount;

    CUDA_CALL(cudaMalloc(&d_kVals, numKs * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_kIndices, numKs * sizeof (uint)));

    for (register int i = 0; i < numKs; i++) {
      kthBucketScanner[i] = 0;
      kIndices[i] = i;
    }

    // variable to store the end result
    int newInputLength, newInputLengthAlt;
    T* newInput;
    T* newInputAlt;

    /// ***********************************************************
    /// **** STEP 1: Initialization 
    /// **** STEP 1.3: Sort the klist
    /// and keep the old index
    /// ***********************************************************

    CUDA_CALL(cudaMemcpy(d_kIndices, kIndices, numKs * sizeof (uint), 
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_kVals, kVals, numKs * sizeof (uint), 
                         cudaMemcpyHostToDevice)); 

    // sort the given indices
    thrust::device_ptr<uint>kVals_ptr(d_kVals);
    thrust::device_ptr<uint>kIndices_ptr(d_kIndices);
    thrust::sort_by_key(kVals_ptr, kVals_ptr + numKs, kIndices_ptr);

    CUDA_CALL(cudaMemcpy(kIndices, d_kIndices, numKs * sizeof (uint), 
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(kVals, d_kVals, numKs * sizeof (uint), 
                         cudaMemcpyDeviceToHost)); 

    int kMaxIndex = numKs - 1;
    int kOffsetMax = 0;
    while (kVals[kMaxIndex] == length) {
      output[kIndices[numKs-1]] = maximum;
      numKs--;
      kMaxIndex--;
      kOffsetMax++;
    }

    int kOffsetMin = 0;
    while (kVals[0] == 1) {
      output[kIndices[0]] = minimum;
      kIndices++;
      kVals++;
      numKs--;
      kOffsetMin++;
    }

    timing(1,1);
    /// ***********************************************************
    /// **** STEP 2: CreateBuckets 
    /// ****  Declare and Generate Pivots and Slopes
    /// ***********************************************************
    timing(0,2);
    // since slopes and pivots will be reused as oldminimums and oldslopes, preallocate to the right size
    uint slopesize = max(numPivots, numKs);
    CUDA_CALL(cudaMalloc(&d_slopes, slopesize * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_pivots, slopesize * sizeof(T)));
    CUDA_CALL(cudaMalloc(&d_pivottree, numPivots * sizeof(T)));

    if (world_rank == 0) {
      // Find bucket sizes using a randomized selection
      generatePivots<T>(pivots, pivottree, slopes, d_vector, length, numPivots, sampleSize, 
                        numBuckets, minimum, maximum);
      SAFEcuda("generatePivots");
      // make any slopes that were infinity due to division by zero (due to no 
      //  difference between the two associated pivots) into zero, so all the
      //  values which use that slope are projected into a single bucket
      for (register int i = 0; i < numPivots - 1; i++)
        if (isinf(slopes[i]))
          slopes[i] = 0;

      // Send slopes, pivots, and pivottree to other processes
      for (int i = 1; i < world_size; i++) {
        switch (datatype) {
        case 0:
          MPI_Send(pivots, numPivots, MPI_FLOAT, i, datatype, MPI_COMM_WORLD); 
          MPI_Send(pivottree, numPivots, MPI_FLOAT, i, datatype, MPI_COMM_WORLD);
          break;
        case 1:
          MPI_Send(pivots, numPivots, MPI_DOUBLE, i, datatype, MPI_COMM_WORLD); 
          MPI_Send(pivottree, numPivots, MPI_DOUBLE, i, datatype, MPI_COMM_WORLD);
          break;
        case 2:
          MPI_Send(pivots, numPivots, MPI_UNSIGNED, i, datatype, MPI_COMM_WORLD); 
          MPI_Send(pivottree, numPivots, MPI_UNSIGNED, i, datatype, MPI_COMM_WORLD);
          break;
        } // end switch
        MPI_Send(slopes, numPivots, MPI_DOUBLE, i, datatype, MPI_COMM_WORLD);
      } // end for
    } // end if (world_rank == 0)

    else {
      MPI_Status status;
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      switch (status.MPI_TAG) {
      case 0:
        MPI_Recv(pivots, numPivots, MPI_FLOAT, 0, datatype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(pivottree, numPivots, MPI_FLOAT, 0, datatype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      case 1:
        MPI_Recv(pivots, numPivots, MPI_DOUBLE, 0, datatype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(pivottree, numPivots, MPI_DOUBLE, 0, datatype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      case 2:
        MPI_Recv(pivots, numPivots, MPI_UNSIGNED, 0, datatype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(pivottree, numPivots, MPI_UNSIGNED, 0, datatype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      }
        MPI_Recv(slopes, numPivots, MPI_DOUBLE, 0, datatype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } // end else

    MPI_Barrier(MPI_COMM_WORLD);

    CUDA_CALL(cudaMemcpy(d_slopes, slopes, numPivots * sizeof(double), 
                         cudaMemcpyHostToDevice));  
    CUDA_CALL(cudaMemcpy(d_pivots, pivots, numPivots* sizeof(T), 
                         cudaMemcpyHostToDevice)); 
    CUDA_CALL(cudaMemcpy(d_pivottree, pivottree, numPivots* sizeof(T), 
                         cudaMemcpyHostToDevice));
    timing(1,2);
    /// ***********************************************************
    /// **** STEP 3: AssignBuckets 
    /// **** Using the function assignSmartBucket
    /// ***********************************************************
    timing(0,3);

    // setToAllZero(d_bucketCount, numBuckets * numBlocks);
    
    // cudaDeviceSynchronize();

    //Distribute elements into their respective buckets
    assignSmartBucket<T><<<numBlocks, threadsPerBlock, 2 * numPivots * sizeof(T) +  
      + numPivots * sizeof(double) + numBuckets * sizeof(uint)>>>
      (d_vector, length, numBuckets, d_slopes, d_pivots, d_pivottree, numPivots, 
       d_elementToBucket, d_bucketCount, offset);

    printf ("\n");

    MPI_Barrier(MPI_COMM_WORLD);

    SAFEcuda("assignSmartBucket");

    CUDA_CALL(cudaMemcpy(slopes, d_slopes, numPivots * sizeof(double), 
                         cudaMemcpyDeviceToHost));  

    CUDA_CALL(cudaMemcpy(h_totalBucketCount, d_bucketCount, totalBucketSize
                         , cudaMemcpyDeviceToHost));
    


    // COMBINE THE COUNTS
    if (world_rank != 0) {
      MPI_Send (h_totalBucketCount, totalBucketLength, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
    }
    else {
      for (int i = 1; i < world_size; i++) {
        MPI_Recv (h_totalBucketCount + (i * totalBucketLength), totalBucketLength, MPI_UNSIGNED
                  , i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    } // end if/else
    
    // Combine entries across h_totalBucketCount to form a single bucketCount array
    if (world_rank == 0) {
      CUDA_CALL(cudaMemcpy(d_totalBucketCount, h_totalBucketCount, totalBucketSize * world_size, cudaMemcpyHostToDevice));
      sumTotalCounts<T><<<numBlocks, threadsPerBlock>>>(d_totalBucketCount, numBlocks, numBuckets, world_size);
      CUDA_CALL(cudaMemcpy(d_bucketCount, d_totalBucketCount, totalBucketSize, cudaMemcpyDeviceToDevice));
    } // end if (world_rank == 0)

    SAFEcuda("sumTotalCounts");

    timing(1,3);
    /// ***********************************************************
    /// **** STEP 4: IdentifyActiveBuckets 
    /// **** Find the kth buckets
    /// **** and update their respective indices
    /// ***********************************************************
    timing(0,4);

    MPI_Barrier(MPI_COMM_WORLD);

    int totalNewInputLength;
    if (world_rank == 0) {
      sumCounts<<<numBuckets/threadsPerBlock, threadsPerBlock>>>(d_bucketCount,          
                                                                 numBuckets, numBlocks);
    
      SAFEcuda("sumCounts");

      findKBuckets(d_bucketCount, h_bucketCount, numBuckets, kVals, numKs, 
                   kthBucketScanner, kthBuckets, numBlocks);
      SAFEcuda("findKBuckets");

      // for (int i = 0; i < numBuckets; i++)
      //   printf ("h_bucketCount[%d] = %d   ", i, h_bucketCount[i]);

      // we must update K since we have reduced the problem size to elements in the 
      // kth bucket.
      //  get the index of the first element
      //  add the number of elements
      uniqueBuckets[0] = kthBuckets[0];
      reindexCounter[0] = 0;
      numUniqueBuckets = 1;
      kVals[0] -= kthBucketScanner[0];

      for (int i = 1; i < numKs; i++) {
        if (kthBuckets[i] != kthBuckets[i-1]) {
          uniqueBuckets[numUniqueBuckets] = kthBuckets[i];
          reindexCounter[numUniqueBuckets] = 
            reindexCounter[numUniqueBuckets-1]  + h_bucketCount[kthBuckets[i-1]];
          printf ("Adding %d at i = %d index = %d\n", h_bucketCount[kthBuckets[i-1]], i, kthBuckets[i-1]);
          numUniqueBuckets++;
        }
        kVals[i] = reindexCounter[numUniqueBuckets-1] + kVals[i] - kthBucketScanner[i];
      }

      totalNewInputLength = reindexCounter[numUniqueBuckets-1] 
        + h_bucketCount[kthBuckets[numKs - 1]];
      printf ("Adding %d at i = %d index = %d\n", h_bucketCount[kthBuckets[numKs-1]], numKs, kthBuckets[numKs-1]);
    } // end if (world_rank == 0)

    if (world_rank == 0) {
      for (int i = 1; i < world_size; i++) {
        MPI_Send(kthBuckets, numKs, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
        MPI_Send(&numUniqueBuckets, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(uniqueBuckets, numUniqueBuckets, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      }
    }
    else {
      MPI_Recv(kthBuckets, numKs, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
      MPI_Recv(&numUniqueBuckets, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(uniqueBuckets, numUniqueBuckets, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // locally calculate newInputLength
    int sumsRowIndex = numBuckets * (numBlocks-1);
    CUDA_CALL(cudaMemcpy(d_localBucketCount, h_totalBucketCount, totalBucketSize, cudaMemcpyHostToDevice));
    sumCounts<<<numBuckets/threadsPerBlock, threadsPerBlock>>>(d_localBucketCount
                                                               , numBuckets, numBlocks);
    CUDA_CALL(cudaMemcpy(h_localBucketCount, d_localBucketCount + sumsRowIndex
                         , sizeof(uint) * numBuckets, cudaMemcpyDeviceToHost));
    localReindexCounter[0] = 0;
    int bucketCounter = 1;
    for (int i = 1; i < numKs; i++) {
      if (kthBuckets[i] != kthBuckets[i-1]) {
        localReindexCounter[bucketCounter] = 
          localReindexCounter[bucketCounter-1] + h_localBucketCount[kthBuckets[i-1]];
        printf ("Summing %d at i = %d index = %d on %d\n", h_localBucketCount[kthBuckets[i-1]], i, kthBuckets[i-1], world_rank);
        bucketCounter++;
      } // end if
    } // end for

    newInputLength = localReindexCounter[numUniqueBuckets-1]
      + h_localBucketCount[kthBuckets[numKs-1]];
    printf ("Summing %d at i = %d index = %d on %d\n", h_localBucketCount[kthBuckets[numKs-1]], numKs, kthBuckets[numKs-1], world_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    // reindex the counts locally and globally
    CUDA_CALL(cudaMalloc(&d_reindexCounter, numKs * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_uniqueBuckets, numKs * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_localReindexCounter, numKs * sizeof(uint)));

    CUDA_CALL(cudaMemcpy(d_reindexCounter, reindexCounter, 
                         numUniqueBuckets * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_uniqueBuckets, uniqueBuckets, 
                         numUniqueBuckets * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_localReindexCounter, localReindexCounter,
                         numUniqueBuckets * sizeof(uint), cudaMemcpyHostToDevice));

    reindexCounts<<<(int) ceil((float)numUniqueBuckets/threadsPerBlock),
      threadsPerBlock>>>(d_localBucketCount, numBuckets, numBlocks, d_localReindexCounter,
                         d_uniqueBuckets, numUniqueBuckets);

    if (world_rank == 0) {
      reindexCounts<<<(int) ceil((float)numUniqueBuckets/threadsPerBlock), 
        threadsPerBlock>>>(d_bucketCount, numBuckets, numBlocks, d_reindexCounter, 
                           d_uniqueBuckets, numUniqueBuckets);
    }
    SAFEcuda("reindexCounts");

    MPI_Barrier(MPI_COMM_WORLD);

    cudaDeviceSynchronize();

    //send the now updated information out to all processes after copying it to the host
    if (world_rank == 0) {
      CUDA_CALL(cudaMemcpy(h_trueBucketCount, d_bucketCount, totalBucketSize, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy(reindexCounter, d_reindexCounter, numKs * sizeof (uint), cudaMemcpyDeviceToHost));
      for (int i = 1; i < world_size; i++) {
        MPI_Send(h_trueBucketCount, numBlocks * numBuckets, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
        MPI_Send(reindexCounter, numKs, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
        MPI_Send(kVals, numKs, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
        MPI_Send(&totalNewInputLength, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
    }
    else {
      MPI_Recv(h_trueBucketCount, numBlocks * numBuckets, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(reindexCounter, numKs, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(kVals, numKs, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&totalNewInputLength, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      CUDA_CALL(cudaMemcpy (d_bucketCount, h_trueBucketCount, totalBucketSize, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy (d_reindexCounter, reindexCounter, numKs * sizeof (uint), cudaMemcpyHostToDevice));
    }

    printf ("total new len = %d, new len = %d on %d\n", totalNewInputLength, newInputLength, world_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    timing(1,4);
    /// ***********************************************************
    /// **** STEP 5: Reduce 
    /// **** Copy the elements from the unique active buckets
    /// **** to a new vector 
    /// ***********************************************************
    timing(0,5);

    // allocate memory for the new array
    if (world_rank != 0)
      CUDA_CALL(cudaMalloc(&newInput, newInputLength * sizeof(T)));
    else
      CUDA_CALL(cudaMalloc(&newInput, totalNewInputLength * sizeof(T)));

    int numUnique_extended = ( 2 << (int)( floor( log2( (float)numUniqueBuckets ) ) ) );
    if (numUnique_extended > numUniqueBuckets+1){
      numUnique_extended--;
    } else {
      numUnique_extended = (numUnique_extended << 1 ) - 1;
    }

    copyElements_tree<T><<<numBlocks, threadsPerBlock, 
      numUnique_extended * sizeof(uint)>>>(d_vector, length, d_elementToBucket, 
                                           d_uniqueBuckets, numUniqueBuckets, numUnique_extended, newInput, offset, 
                                           d_localBucketCount, numBuckets);
    SAFEcuda("copyElements");

    
    timing(1,5);
    /// ***********************************************************
    /// **** STEP 6: sort&choose
    /// **** Using thrust::sort on the reduced vector and the
    /// **** updated indices of the order statistics, 
    /// **** we solve the reduced problem.
    /// ***********************************************************
    timing(0,6);
    

    /*


    // declare and allocate device memory for recursion phase
    uint *d_numUniquePerBlock, *d_markedBucketFlags, *d_sums, *d_Kbounds, *d_oldReindexCounter;
    CUDA_CALL(cudaMalloc(&d_numUniquePerBlock, (numKs) * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_markedBucketFlags, numKs * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_sums, numKs * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_Kbounds, (numKs+1) * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_oldReindexCounter, numKs * sizeof(uint)));

    // I'm not sure if new Minimums needs to be double or not.
    // if not, we can also declare old minimums and use d_pivots
    double * d_newMinimums;
    //    T * d_newMinimums;
    double * d_newSlopes, *d_oldSlopes;
    CUDA_CALL(cudaMalloc(&d_newMinimums, numKs * sizeof(double)));
    //    CUDA_CALL(cudaMalloc(&d_newMinimums, numNewActive * sizeof(T)));
    CUDA_CALL(cudaMalloc(&d_newSlopes, numKs * sizeof(double)));
    d_oldSlopes = d_slopes;

    uint* tempReindex;

    CUDA_CALL(cudaMalloc (&newInputAlt, sizeof(T) * newInputLength));

    // declare and initialize parameters for recursion
    uint numOldActive, numNewActive, oldNumSmallBuckets, newNumSmallBuckets;
    numOldActive = numPivots - 1;
    numNewActive = numUniqueBuckets;
    oldNumSmallBuckets = numBuckets/numOldActive;


    int recreateThreads = 128;

    // *****************************************************
    // Here seems to be where we begin the recursion
    // *****************************************************

    // determine the number of buckets per new block
    newNumSmallBuckets = numBlocks*numBuckets/numNewActive;
    //    newNumSmallBuckets = numBuckets/numNewActive;

    CUDA_CALL(cudaMemcpy (d_kVals, kVals, numKs * sizeof (uint), cudaMemcpyHostToDevice));

    int recreateBlocks = numNewActive/recreateThreads + 1;

    // Recreate sub-buckets

    printf ("about to recreate on %d\n", world_rank);

    // Recreate buckets on each processor
    recreateBuckets<T><<<recreateBlocks, recreateThreads
      , numOldActive*sizeof(double)*2>>>(d_uniqueBuckets, d_newSlopes, d_newMinimums
                                         , numNewActive, d_oldSlopes, d_pivots, numOldActive
                                         , oldNumSmallBuckets, newNumSmallBuckets);

    cudaDeviceSynchronize();
    SAFEcuda("recreateBuckets");

    reassignBuckets<T><<<numNewActive, threadsPerBlock
      , newNumSmallBuckets*sizeof(uint)>>>(newInput, newInputLength, d_reindexCounter, d_newSlopes
                                           , d_newMinimums, numNewActive, newNumSmallBuckets
                                           , d_elementToBucket, d_bucketCount);

    cudaDeviceSynchronize();
    SAFEcuda("reassignBuckets");

    // Need to combine information again

    tempReindex = d_oldReindexCounter;
    d_oldReindexCounter = d_reindexCounter;
    d_reindexCounter = tempReindex;
    numOldActive = numNewActive;

    cudaThreadSynchronize();
    
    //    timing(1,6);
     
    SAFEcuda("pre findKBuckets");


    // compute Kbounds and copy to device
    uint blockOffset;
    uint Kbounds[numKs+1];
    Kbounds[0]=0;
    uint j = 0;
    uint i = 1;
    uint k = kVals[j];
    while (i < numOldActive) {
      blockOffset = reindexCounter[i];  // in recursion, we need to copy from the d_reindexsums
      while ( (k <= blockOffset) && (j < numKs) ) {
        j++;
        k = kVals[j];
      }
      Kbounds[i]=j;
      i++;
    }
    Kbounds[numOldActive]=numKs;

    CUDA_CALL(cudaMemcpy(d_Kbounds, Kbounds, 
                         (numOldActive+1) * sizeof(uint), cudaMemcpyHostToDevice));
    // *******************************

    newInputLengthAlt = findKbucketsByBlock (d_bucketCount, d_oldReindexCounter, d_Kbounds, d_reindexCounter
                                             , d_sums, d_kVals, d_markedBucketFlags, d_numUniquePerBlock
                                             , d_uniqueBuckets, &numNewActive, numOldActive, newNumSmallBuckets, numKs);

    cudaThreadSynchronize();


    CUDA_CALL(cudaMemcpy(uniqueBuckets, d_uniqueBuckets, 
                         numNewActive * sizeof(uint), cudaMemcpyDeviceToHost));
    //    for (int i=0; i<numNewActive; i++) printf("active[%d]=%d\n",i,uniqueBuckets[i]);

    SAFEcuda("findKBucketsByBlock");

    

    cudaThreadSynchronize();
    
    int reducedlength = Reduction<T>(newInput, newInputAlt, d_elementToBucket, d_uniqueBuckets, d_oldReindexCounter
                                     , d_numUniquePerBlock, newInputLength, numOldActive, numNewActive);



    SAFEcuda("Reduction");

    */

    // OLD STUFF BEGINS
    timing(1,6);
    timing(0,7);
    //printf ("NEW LENGTH = %d\n", newInputLengthAlt);
    
    T* totalNewInput;
    if (world_rank == 0)
      totalNewInput = (T*) malloc (totalNewInputLength * sizeof(T));
    int recvCounts[world_size];
    int recvDispls[world_size];

    //printf ("SENDING\n");
    if (world_rank != 0)
      MPI_Send(&newInputLength, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    else {
      for (int i = 1; i < world_size; i++)
        MPI_Recv(recvCounts + i, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      recvCounts[0] = newInputLength;
      recvDispls[0] = 0;
      for (int i = 1; i < world_size; i++)
        recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];
    }

    T* h_newInput;
    h_newInput = (T*) malloc (newInputLength * sizeof(T));

    if (world_rank != 0)
      CUDA_CALL(cudaMemcpy (h_newInput, newInput, newInputLength * sizeof(T), cudaMemcpyDeviceToHost));
    else
      CUDA_CALL(cudaMemcpy (totalNewInput, newInput, newInputLength * sizeof(T), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    // for (int i = 0; i <newInputLength; i++)
    //   printf ("h_newInput[%d] on %d = %f\n", i, world_rank, h_newInput[i]);
    
    // Combine the reduced vectors into a single input vector for sorting
    if (world_rank != 0) {
      switch (datatype) {
      case 0: 
        MPI_Send(h_newInput, newInputLength, MPI_FLOAT, 0, datatype, MPI_COMM_WORLD);
        break;
      case 1:
        MPI_Send(h_newInput, newInputLength, MPI_DOUBLE, 0, datatype, MPI_COMM_WORLD);
        break;
      case 2:
        MPI_Send(h_newInput, newInputLength, MPI_UNSIGNED, 0, datatype, MPI_COMM_WORLD);
        break;
      }
    }

    else {
      //MPI_Status status;
      for (int i = 1; i < world_size; i++) {
        switch(datatype) {  
        case 0:
          MPI_Recv(totalNewInput + recvDispls[i], recvCounts[i], MPI_FLOAT, i
                   , datatype, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
          break;
        case 1:
          MPI_Recv(totalNewInput + recvDispls[i], recvCounts[i], MPI_DOUBLE, i
                   , datatype, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
          break;
        case 2:
          MPI_Recv(totalNewInput + recvDispls[i], recvCounts[i], MPI_UNSIGNED, i
                   , datatype, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
          break;
        }
      }
      CUDA_CALL(cudaMemcpy(newInput, totalNewInput, totalNewInputLength * sizeof(T), cudaMemcpyHostToDevice));
    }

    T* d_totalNewInput;

    // Sort and choose
    if (world_rank == 0) {
      for (int i = 0; i < totalNewInputLength; i++)
        printf ("totalNewInput[%d] = %f\n", i, totalNewInput[i]);
      for (int i = 0; i < numKs; i++)
        printf ("kVals[%d] = %d\n", i, kVals[i]);
      CUDA_CALL(cudaMalloc(&d_totalNewInput, totalNewInputLength * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_totalNewInput, totalNewInput, totalNewInputLength * sizeof(T), cudaMemcpyHostToDevice));
      sort_phase<T>(d_totalNewInput, totalNewInputLength);
      SAFEcuda("sort_phase");
      T * d_output = (T *) d_elementToBucket;
      CUDA_CALL(cudaMemcpy (d_output, output, 
                            (numKs + kOffsetMin + kOffsetMax) * sizeof (T), 
                            cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpy (d_kIndices, kIndices, numKs * sizeof (uint), 
                            cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy (d_kVals, kVals, numKs * sizeof (uint), cudaMemcpyHostToDevice));

      copyValuesInChunk<T><<<numBlocks, threadsPerBlock>>>(d_output, d_totalNewInput, d_kVals, d_kIndices, numKs);
    
      SAFEcuda("copyValuesInChunk");

      CUDA_CALL(cudaMemcpy (output, d_output, 
                            (numKs + kOffsetMin + kOffsetMax) * sizeof (T), 
                            cudaMemcpyDeviceToHost));
    }
    
    SAFEcuda("sort_phase global");

    //free all used memory
    cudaFree(d_pivots);
    cudaFree(d_pivottree);
    cudaFree(d_slopes); 
    free(h_bucketCount);
    cudaFree(d_bucketCount);
    cudaFree(d_uniqueBuckets); 
    //cudaFree(d_markedBucketFlags); 
    //cudaFree(d_sums); 
    cudaFree(d_reindexCounter);
    //cudaFree(d_oldReindexCounter);
    //cudaFree(d_newMinimums);
    //cudaFree(d_newSlopes);
    cudaFree(newInput); 
 
  
    cudaFree(d_elementToBucket);  
    cudaFree(d_kIndices); 
    cudaFree(d_kVals); 
    //cudaFree(newInputAlt);
    free (kIndices - kOffsetMin);

    /// ***********************************************************
    /// **** STEP 6: Recurse
    /// **** Using thrust::sort on the reduced vector and the
    /// **** updated indices of the order statistics, 
    /// **** 
    /// ***********************************************************
    // timing(0,6)
  


    timing(1,7);
    return 1;
  }


  /* Wrapper function around the multi-selection fucntion that inverts the given k indices.
   */
  template <typename T>
  T bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, int numKs
                              , T * outputs, int blocks, int threads, int world_rank
                              , int world_size, char* processor_name, uint datatype, uint origLen) { 

    int numBuckets = 8192;
    uint kVals[numKs];


    // turn it into kth smallest
    for (register int i = 0; i < numKs; i++) 
      kVals[i] = origLen - kVals_ori[i] + 1;
   
    bucketMultiSelect<T> (d_vector, length, kVals, numKs, outputs, blocks, threads, numBuckets, 17, world_rank, world_size, processor_name, datatype);

    return 1;
  }
}

