
  template <typename T>
  __global__ void copyElements_recurse (T* d_vector, int length, uint* elementToBucket
                                        , uint * uniqueBuckets, T* newArray, int numBlocks
                                        , uint * d_bucketCount, int numBuckets, int* blockBounds
                                        , int newLength, int sumsRowIndex, uint* activeBucketCounts) {


    // OLD BST VERSION IS IN /MAP/testFile.cu

    // Calculate blockStart and blockEnd, the number of buckets created by blocks up through
    //   blockIdx.x - 1, and the number of buckets create by blocks through blockIdx.x, respectively
    int blockStart = blockBounds[blockIdx.x];
    int blockEnd = blockBounds[blockIdx.x + 1] - 1;
    __shared__ int numUniqueBlock;
    numUniqueBlock = blockEnd - blockStart + 1;
    uint bucketSize = activeBucketCounts[blockIdx.x];

    extern __shared__ uint sharedBuckets[];
  
    // Read relevant unique buckets for a given block into shared memory
    if (threadIdx.x < numUniqueBlock)
      for (int i = 0; i < numUniqueBlock; i += blockDim.x) {
        sharedBuckets[i + threadIdx.x] = uniqueBuckets[i + threadIdx.x + blockStart];
      }
    syncthreads ();

    int idx = threadIdx.x;
    int temp, min, max, mid, compare;
  
    // For each element, binary search through active buckets to see if the element is relevant.
    //  If it's in an active bucket, copy to a new array.
    if (idx < bucketSize) {
      for (int i = idx; i < bucketSize; i += blockDim.x) {
        //printf ("block: %d\n", blockIdx.x);
        temp = elementToBucket[i];
        min = 0;
        max = blockEnd - blockStart;
        compare = 0;
	//printf("\n 2 \n");      
        // Binary search
        for (int j = 1; j < numUniqueBlock; j *= 2) {
          mid = (max + min) / 2;
          compare = temp > sharedBuckets[mid];
          min = compare ? mid : min;
          max = compare ? max : mid;
          //printf ("block: %d, max = %d\n", blockIdx.x, max);
        } //end for
        syncthreads();
        // CHECK LOOP INVARIANT: USE MIN BELOW? < OR <= IN FOR LOOP?
        if (sharedBuckets[max] == temp) {
          //printf ("index = %d, max = %d\n", i, uniqueBuckets[max]);
          //int k = atomicDec(d_bucketCount + temp + sumsRowIndex, newLength) - 1;
          newArray[atomicDec(d_bucketCount + temp + sumsRowIndex, newLength) - 1] = d_vector[i];
          //newArray[k] = d_vector[i];
          //printf ("index = %d, val = %f, block = %d\n", k, newArray[k], blockIdx.x);
        } // end if (uniqueBuckets[max] == temp)
      } //end for
    } // end if (idx < length)
  }  // ends copyElements_recurse kernel


