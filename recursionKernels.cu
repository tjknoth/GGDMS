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
template <typename T>
__global__ void recreateBuckets (uint * d_uniqueBuckets, 
                                 double * newSlopes, double * newMinimums, const uint numNewActive,
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
  double * shared_oldMinimums = (double *)&shared_oldSlopes[numOldActive];

  // Read the oldSlopes and oldMinimums into shared memory
  for (int i = threadIndex; i < numOldActive; i += numThreadsPerBlock) {
    shared_oldMinimums[i] = (double)oldMinimums[i];
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
                                 double * newSlopes, double * newMinimums, const uint numNewActive,
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
    localBucket = (int) (((double)num - minimum) * slope);
    if (localBucket == newNumSmallBuckets) {
      localBucket = newNumSmallBuckets-1;  // ensure local bucket stays within this block
    } 
       
    // assign this element to the correct bucket
    elementToBucket[i] = localBucket + blockBucketOffset;
    // increment the local counter
    atomicInc(counts + localBucket, vecLength); 

  } // end for i=blockStart ...

  syncthreads();

  // Copy the local counts to the global device d_bucketCount with appropriate offset
  for (int i = threadIndex; i < newNumSmallBuckets; i+=numThreadsPerBlock) {
    bucketCount[i + blockBucketOffset] = counts[i];
  }  // end for loop on counts copy to global device memory

} // end kernel reassignBuckets




