// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>


/* Kernel to create new minimums and new slopes which define the new buckets
 *
 * Notes: Launch this kernel with a fixed numThreadsPerBlock, probably 128,
 *        and the number of blocks computed to ensure you have numUniqueBuckets threads.
 *        Launch the kernel with shared memory allocated for
 *        size = numOldActive*(sizeof(T) + sizeof(double))
 */
template <typename T>
__global__ void recreateBuckets (uint * d_uniqueBuckets, 
                                 double * newSlopes, T * newMinimums, const uint numNewActive,
                                 double * oldSlopes, T * oldMinimums, const uint numOldActive,
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
  T * shared_oldMinimums = (T *)&shared_oldSlopes[numOldActive];

  // Read the oldSlopes and oldMinimums into shared memory
  //    int readIndex;
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
    newSlopes[index] = newNumSmallBuckets * oldSlope;

  } // end if(index<numUniqueBuckets)
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
                                 double * newSlopes, T * newMinimums, const uint numNewActive,
                                 const uint newNumSmallBuckets, uint * elementToBucket,
                                 uint * bucketCount)
{
  int localBucket;
  int numThreadsPerBlock = blockDim.x;
  int threadIndex = threadIdx.x;
  int blockIndex = blockIdx.x;
  T num;
  
  // Since slope, minimum, and blockOffset are the same for every element in 
  // the same block, copy them to shared memory for fast access
  __shared__ double slope;
  __shared__ T minimum;
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
  }  // end if threadIndex<1 for shared memory constants
  
  // declare shared memory counter as counts
  extern __shared__ uint counts[];
   
  // Initialize counts to zero
  for (int i = threadIndex; i < newNumSmallBuckets; i+=numThreadsPerBlock) {
    counts[i] = 0;
  }  // end for loop on counts initialization

  syncthreads();

  // assign elements in this block to appropriate buckets
  for (int i = blockStart + threadIndex; i < blockEnd; i += numThreadsPerBlock) {

    // read in the value from the current vector
    num = vector[i];

    // compute the local bucket via the linear projection
    localBucket = (int) (((double)num - (double)minimum) * slope);
    // ensure the local bucket is not out of bounds
    // if (localBucket > newNumSmallBuckets - 1) {
    //    localBucket = newNumSmallBuckets-1;
    // }
    if (localBucket == newNumSmallBuckets) {
      localBucket = newNumSmallBuckets-1;
    } 
       
    // assign this element to the correct bucket
    elementToBucket[i] = localBucket + blockBucketOffset;
    // increment the local counter
    atomicInc(counts + localBucket, newNumSmallBuckets); 

  } // end for i=blockStart ...

  // **********
  /*
    syncthreads();

    for (int i = blockStart + threadIndex; i < blockEnd; i += numThreadsPerBlock) {
    printf("Blk %d, thd %d, buck %d\n", blockIdx.x, threadIdx.x, elementToBucket[i]);
    }
  */
  // ***********

  // Copy the local counts to the global device d_bucketCount with appropriate offset
  syncthreads();
  for (int i = threadIndex; i < newNumSmallBuckets; i+=numThreadsPerBlock) {
    bucketCount[i + blockBucketOffset] = counts[i];
    // *******
    //if (counts[i]>0) printf("Block %d, thread %d, count %d\n" , blockIdx.x, threadIdx.x, counts[i]);
    // ********
  }  // end for loop on counts copy to global device memory

} // end kernel reassignBuckets





  // template <typename T>
  // __global__ void copyElements_recurse (T* d_vector, int length, uint* elementToBucket
  //                                       , uint * uniqueBuckets, T* newArray, int numBlocks
  //                                       , uint * d_bucketCount, int numBuckets, int* blockBounds
  //                                       , int newLength, uint* activeBucketCounts) {



  //   // Calculate blockStart and blockEnd, the number of buckets created by blocks up through
  //   //   blockIdx.x - 1, and the number of buckets create by blocks through blockIdx.x, respectively
  //   int blockStart = blockBounds[blockIdx.x];
  //   int blockEnd = blockBounds[blockIdx.x + 1] - 1;
  //   __shared__ int numUniqueBlock;
  //   numUniqueBlock = blockEnd - blockStart + 1;
  //   uint bucketSize = activeBucketCounts[blockIdx.x];

  //   extern __shared__ uint sharedBuckets[];
  
  //   // Read relevant unique buckets for a given block into shared memory
  //   if (threadIdx.x < numUniqueBlock)
  //     for (int i = 0; i < numUniqueBlock; i += blockDim.x) {
  //       sharedBuckets[i + threadIdx.x] = uniqueBuckets[i + threadIdx.x + blockStart];
  //     }
  //   syncthreads ();

  //   int idx = threadIdx.x;
  //   int temp, min, max, mid, compare;
  
  //   // For each element, binary search through active buckets to see if the element is relevant.
  //   //  If it's in an active bucket, copy to a new array.
//   if (idx < bucketSize) {
//     for (int i = idx; i < bucketSize; i += blockDim.x) {
//       temp = elementToBucket[i];
//       min = 0;
//       max = blockEnd - blockStart;
//       compare = 0;  
//       // Binary search
//       for (int j = 1; j < numUniqueBlock; j *= 2) {
//         mid = (max + min) / 2;
//         compare = temp > sharedBuckets[mid];
//         min = compare ? mid : min;
//         max = compare ? max : mid;
//       } //end for
//       syncthreads();
//       if (sharedBuckets[max] == temp) {
//         //printf ("index = %d, max = %d\n", i, uniqueBuckets[max]);
//         //int k = atomicDec(d_bucketCount + temp, newLength) - 1;
//         newArray[atomicDec(d_bucketCount + temp, newLength) - 1] = d_vector[i];
//         //newArray[k] = d_vector[i];
//         //printf ("index = %d, val = %f, block = %d\n", k, newArray[k], blockIdx.x);
//       } // end if (uniqueBuckets[max] == temp)
//     } //end for
//   } // end if (idx < length)
// }  // ends copyElements_recurse kernel


template <typename T>
__global__ void copyElements_recurse (T* d_vector, T* newArray, int length, uint* elementToBucket
                                      , int numBuckets, int numBlocks, uint* d_bucketCount
                                      , uint* oldReindexCounter, uint* d_uniqueBuckets) {

  __shared__ uint elementsPerBlock;
  __shared__ uint blockOffset;
  int bucketsPerBlock;
 
  extern __shared__ uint blockActiveBuckets[];

  if (threadIdx.x < 1) {
    int i, j = 0;
    int firstBucket = numBuckets * blockIdx.x / numBlocks;
    int lastBucket = (numBuckets * (blockIdx.x + 1) / numBlocks) - 1;
    blockOffset = oldReindexCounter[blockIdx.x];
    if (blockIdx.x + 1 < numBlocks)
      elementsPerBlock = oldReindexCounter[blockIdx.x + 1] - blockOffset;
    else
      elementsPerBlock = length - blockOffset;
    for (i = 0; d_uniqueBuckets[i] < firstBucket; i++);
    while (d_uniqueBuckets[i] <= lastBucket) {
      blockActiveBuckets[j] = d_uniqueBuckets[i];
      i++;
      j++;
    } //end while
    bucketsPerBlock = j;
  } //end if (threadIdx.x < 1)

  syncthreads();

  int temp, min, max, mid, compare;

  for (int i = threadIdx.x; i < elementsPerBlock; i += blockDim.x) {
    uint index = i + blockOffset;
    temp = elementToBucket[index];
    min = 0;
    max = bucketsPerBlock;
    compare = 0;  
    // Binary search
    for (int j = 1; j < bucketsPerBlock + 1; j *= 2) {
      mid = (max + min) / 2;
      compare = temp > blockActiveBuckets[mid];
      min = compare ? mid : min;
      max = compare ? max : mid;
    } //end for

    if (temp == blockActiveBuckets[max]) {
      //int k = atomicDec(d_bucketCount + temp + blockIdx.x * numBuckets, length) - 1;
      newArray[atomicDec(d_bucketCount + temp + blockIdx.x * numBuckets, length) - 1] = d_vector[index];
      //newArray[k] = d_vector[index];
      //printf ("newArray[%d] = %f\n", k, d_vector[index]);
    } //end if
  } //end for
} //end kernel









/* 
********  OLD STUFF THAT MIGHT BE USEFUL *******
*/

/*
// Initialize counts to zero
for (int i = 0; i < (newNumSmallBuckets / numThreadsPerBlock); i++) {
readIndex = i*numThreadsPerBlock + threadIndex;
if (readIndex < newNumSmallBuckets) {
counts[readIndex] = 0;
} // end if readIndex
}  // end for loop on counts initialization
*/
