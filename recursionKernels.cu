// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>

/* Kernel to create new minimums and new slopes which define the new buckets
 *
 * Notes: Launch this kernel with a fixed numThreadsPerBlock, probably 32,
 *        and the number of blocks computed to ensure you have numUniqueBuckets threads.
 *        Launch the kernel with shared memory allocated for
 *        size = 2*numOldActive*sizeof(double)
 */

__device__ void deviceTag(int marker){
  if (threadIdx.x + blockIdx.x == 0) {
    printf("Tag %d\n",marker);
  }
  syncthreads();
}

template <typename T>
__global__ void printInput(T * vector,int length){
  for (int i = threadIdx.x; i < length; i += blockDim.x) {
    printf("index %d -> %lf\n",i,vector[i]);
  }
  syncthreads();
}

template <typename T>
__global__ void recreateBuckets (uint * d_uniqueBuckets,
                                 double * newSlopes, double * newMinimums, const uint numNewActive,
                                 double * oldSlopes, double * oldMinimums, const uint numOldActive,
                                 const uint oldNumSmallBuckets, const uint newNumSmallBuckets)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int threadIndex = threadIdx.x;
  int numThreadsPerBlock = blockDim.x;
  uint oldBucket;
  uint oldBigBucket;
  uint precount;
  double oldSlope;
  double width;

  // Allocate shared memory pointers
  extern __shared__ uint array[];
  double * shared_oldSlopes = (double *)array;
  double * shared_oldMinimums = (double *)&shared_oldSlopes[numOldActive];

  // Read the oldSlopes and oldMinimums into shared memory
  for (int i = threadIndex; i < numOldActive; i += numThreadsPerBlock) {
    shared_oldMinimums[i] = oldMinimums[i];
    shared_oldSlopes[i] = oldSlopes[i];
  } // end for

  syncthreads();

  if (index < numNewActive) {
    oldBucket = d_uniqueBuckets[index];
    oldBigBucket = oldBucket/oldNumSmallBuckets;
    precount = oldBigBucket * oldNumSmallBuckets;
    oldSlope = shared_oldSlopes[oldBigBucket];
    width = 1 / oldSlope;
    newMinimums[index] = (oldBucket - precount) * width + shared_oldMinimums[oldBigBucket];
    //printf ("newMins[%d] = %.12lf   ", index, (oldBucket - precount) * width + shared_oldMinimums[oldBigBucket]);
    newSlopes[index] = newNumSmallBuckets * oldSlope;
    //printf ("newSlopes[%d] = %f     ", index, newNumSmallBuckets * oldSlope);
  } // end if(index<numUniqueBuckets)
  //if (threadIdx.x < 1) printf("\n********************************\n********************************\n\n");
} // end kernel recreateBuckets






/* kernel which uses the new slopes and minimums defining the new buckets and
 * assigns the elements in a given old bucket to the new buckets
 *
 * Notes: Launch this kernel with one block per new active bucket and max threads per block.
 *        Launch the kernel with shared memory allocated for
 *        size = newNumSmallBuckets*sizeof(uint)
 */
template <typename T>
__global__ void reassignBuckets (T * vector, const int vecLength, uint * bucketBounds,
                                 double * newSlopes, double * newMinimums, const uint numNewActive,
                                 const uint newNumSmallBuckets, uint * elementToBucket,
                                 uint * bucketCount)
{
  int localBucket;
  int numThreadsPerBlock = blockDim.x;
  int threadIndex = threadIdx.x;
  int blockIndex = blockIdx.x;
  T num;



  //deviceTag(2);
  syncthreads();
  // declare shared memory counter as counts
  extern __shared__ uint counts[];

  // Initialize counts to zero
  for (int i = threadIndex; i < newNumSmallBuckets; i+=numThreadsPerBlock) {
    counts[i] = 0;
  }  // end for loop on counts initialization

  syncthreads();

  //deviceTag(1);

  // Since slope, minimum, and blockOffset are the same for every element in
  // the same block, copy them to shared memory for fast access

  __shared__ double slope;
  __shared__ double minimum;
  __shared__ int blockBucketOffset;
  __shared__ int blockStart;
  __shared__ int blockEnd;

  if (threadIndex < 1) {
    slope = newSlopes[blockIndex];
    minimum = newMinimums[blockIndex];
    blockBucketOffset = (blockIndex * newNumSmallBuckets);
    blockStart = bucketBounds[blockIndex];
    if (blockIndex < (numNewActive-1) ) {
      blockEnd = bucketBounds[blockIndex+1];
    } else {
      blockEnd = vecLength;
    }
    //		printf("\n******\n blockIdx.x = %d\n******\nslope = %lf\tminimum = %lf\tblockBucketOffset = %d\tblockStart = %d\tblockEnd = %d\n"
    //			,blockIdx.x,slope,minimum,blockBucketOffset,blockStart,blockEnd);

  }  // end if threadIndex<1 for shared memory constants

  syncthreads();

  /*

    deviceTag(2);
    syncthreads();
    // declare shared memory counter as counts
    extern __shared__ uint counts[];

    // Initialize counts to zero
    for (int i = threadIndex; i < newNumSmallBuckets; i+=numThreadsPerBlock) {
    counts[i] = 0;
    printf("i = %d\n",i);
    }  // end for loop on counts initialization

    syncthreads();
  */

  // assign elements in this block to appropriate buckets
  for (int i = blockStart + threadIndex; i < blockEnd; i += numThreadsPerBlock) {

    // read in the value from the current vector
    num = vector[i];
    /*
      if (num < minimum) {
      printf("num %d = %lf \t minimum = %lf \t block = %d	blockStart = %d	blockEnd = %d\n",i,num,minimum,blockIndex,blockStart,blockEnd);
      }
    */

    // compute the local bucket via the linear projection
    localBucket = (int) (((double)num - minimum) * slope);
    if (localBucket > newNumSmallBuckets - 1) {
      //printf("bucket = %d index %d start %d end %d \nnum %d = %.15f min = %.15f \n\n",localBucket,blockIndex,blockStart,blockEnd,i,num,minimum);
      //   double biggest = newNumSmallBuckets/slope+minimum;
      //   printf("localBucket = %d, newNumSmallBuckets = %d    num = %f   minimum = %f  slope =%f maximum = %f\n",localBucket,newNumSmallBuckets,num,minimum,slope,biggest);
    }
    if (localBucket > newNumSmallBuckets - 1) {
      localBucket = newNumSmallBuckets-1;  // ensure local bucket stays within this block
    }

    // assign this element to the correct bucket
    elementToBucket[i] = localBucket + blockBucketOffset;
    // increment the local counter
    atomicInc(counts + localBucket, vecLength);

  } // end for i=blockStart ...

  syncthreads();

  //printf ("newNumSmallBuckets = %d\n\n\n", newNumSmallBuckets);
  // Copy the local counts to the global device d_bucketCount with appropriate offset
  for (int i = threadIndex; i < newNumSmallBuckets; i+=numThreadsPerBlock) {
    //printf ("copying to %d from %d on block %d   ", i + blockBucketOffset, i, blockIndex);
    bucketCount[i + blockBucketOffset] = counts[i];
  }  // end for loop on counts copy to global device memory

  syncthreads();
} // end kernel reassignBuckets

template <typename T>
void checkBuckets(T * newInput, uint * Kbounds, uint * reindexCounter, uint numOldActive, int numKs, int newInputLength, int numNewActive) {

  // Initialize
  int blockNumKs;
  int start, end;
  int count = 0;




  for (int index = 0; index < numOldActive; index++){
    //	while (index < numOldActive - 1){
    // Find blockNumKs
    if (index < numOldActive - 1) {
      blockNumKs = Kbounds[index + 1] - Kbounds[index];
    } else {
      blockNumKs = numKs - Kbounds[index];
    }


    if (blockNumKs > 1) {
      // Find relevant section of newInput
      start = reindexCounter[index + count];
      //			if (index + count + blockNumKs < numKs) {
      if (index + count + blockNumKs < numOldActive - 1) {
        end = reindexCounter[index + count + blockNumKs];
      } else {
        printf("index %d \t count %d \t blockNumKs %d \n",index,count,blockNumKs);
        end = newInputLength;
      }



      printf("index %d \t blockNumKs = %d \t count = %d \t start = %d \t end = %d \n",index,blockNumKs,count,start,end);

      // Sort
      cubDeviceSort<T>(newInput + start, end - start);


      count += blockNumKs - 1;

    }

  }
}


template <typename T>
__global__ void correctBuckets(uint * d_numUniquePerBlock, T * newInput, T * newInputAlt, uint * bucketBounds, int numOldActive, int numNewActive, double * minimums){



  __shared__ int blockNumKs;
  __shared__ int start;
  __shared__ int end;


  if(blockIdx.x < numOldActive - 1) {
    if (threadIdx.x < 1){
      blockNumKs = d_numUniquePerBlock[blockIdx.x + 1] - d_numUniquePerBlock[blockIdx.x];
      printf("block = %d \t blockNumKs = %d \n",blockIdx.x,blockNumKs);
    }
  } else {
    if (threadIdx.x < 1) {
      blockNumKs = numNewActive - d_numUniquePerBlock[blockIdx.x];
      printf("block = %d \t blockNumKs = %d \n",blockIdx.x,blockNumKs);
    }
  }
  syncthreads();

  if (blockNumKs > 1){

    if (threadIdx.x < 1) {
      start = bucketBounds[blockIdx.x];
      end = bucketBounds[blockIdx.x + blockNumKs];
      printf("block = %d \t numKs = %d \t start = %u \t end = %u \n",blockIdx.x,blockNumKs,start,end);
      //					d_printInput(newInput,start,end);
      //d_sort(newInput + start,end - start);
    }

    syncthreads();






  }
}





__global__ void printMinimums(double * minimums, int numKs){

  if (threadIdx.x + blockIdx.x == 0) {
    for(int i = 0; i < numKs; i ++){
      printf("minimum[%d]=%lf\n",i,minimums[i]);
    }
  }
}

template <typename T>
void sort(T* vector, int length){

  int j = 0;
  while (j < length - 1) {
    for (int i = j + 1; i < length; i++){
      if (vector[i] < vector[j]) {
        T temp = vector[i];
        vector[i] = vector[j];
        vector[j] = temp;
      }
    }
    j++;
  }
}



template <typename T>
__device__ void d_sort(T* vector, int length){

  int j = 0;
  while (j < length - 1) {
    for (int i = j + 1; i < length; i++){
      if (vector[i] < vector[j]) {
        T temp = vector[i];
        vector[i] = vector[j];
        vector[j] = temp;
      }
    }
    j++;
  }
}

template <typename T>
__device__ void d_printInput( T * newInput, int start, int end) {
  printf("\n *** *** *** *** *** \n");
  printf("start = %d \t end = %d\n",start,end);
  for (int i = start; i < end; i+=2) {
    printf("newInput[%d] = %.10lf \t newInput[%d] = %.10lf from device\n",i,newInput[i],i+1,newInput[i+1]);
  }
  printf("\n *** *** *** *** *** \n");
}

template <typename T>
__global__ void checkInput(T* newInput, int newInputLength, int tag){
  for(int i = threadIdx.x; i < newInputLength; i += blockDim.x){
    if (newInput[i] < 0.00000001) printf("Tag %d \t newInput[%d] = 0\n",tag,i);
  }
}
