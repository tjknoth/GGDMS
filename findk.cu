// -*- c++ -*-

  template<typename T>
  void mmsetToAllZero (T * d_vector, int length) {
    cudaMemset(d_vector, 0, length * sizeof(T));
  }


/*
 * This device function will be called by a kernel with each thread finding the 
 * k buckets for each oldActiveBucket (i.e. each block used to reassign buckets) 
*/
__device__ uint d_findKBucketsByBlock(uint * d_bucketCount, uint * kVals, uint * markedBuckets, 
                                    uint * sums, uint * reindexsums, const int numNewSmallBuckets, 
                                    const int blockBucketOffset, const int blockStart,  
                                    const int blockNumKs, const int blockKsOffset, uint * markedBucketFlags)
 {
    int kBucket = blockBucketOffset;
    int blockMaxBucket = blockBucketOffset + numNewSmallBuckets;
    int k;
    int sum = blockStart;
    uint temp;
    int numUniqueBlock=1;
    markedBucketFlags[blockKsOffset] = 1;

    // find the buckets which contain the kVals
    for(register int i = 0; i < blockNumKs; i++) {
      k = kVals[blockKsOffset + i];
      while ((sum < k) & (kBucket < blockMaxBucket)) {
        temp = d_bucketCount[kBucket];
        sum += temp;     
        kBucket++;
      } // end while
      markedBuckets[blockKsOffset + i] = kBucket-1;
      sums[blockKsOffset + i] = sum - temp; 
      reindexsums[blockKsOffset + i] = temp; 

     // determine if this marked bucket is unique
     //  if so, increase the unique counter and create a flag for uniqueness
     //  if not, ensure the flag is 0 and remove the count from reindexsums 
     //          in order to have an accurate cummulative sum outside this kernel
     if (i>0) {
      if ( (markedBuckets[blockKsOffset + i] != markedBuckets[blockKsOffset + i - 1]) ) {
         numUniqueBlock++;
         markedBucketFlags[blockKsOffset + i] = 1;
      } else { 
         markedBucketFlags[blockKsOffset + i] = 0;
         reindexsums[blockKsOffset + i - 1] = 0; 
      }  // end if-else
     } // if buckets are equal

    } // end for

    return numUniqueBlock;

 } // end device function d_findKBucketsByBlock

/*
 * The kernel will launch one thread per old active bucket to get the new k buckets
 * which should be identified as active.  It also gets the sum of all elements that 
 * were in previous new buckets in order to update the desired order statistics.
 * Launch this kernel with a fixed number of threads per block, probably 64, and enough 
 * blocks to achieve numOldActive total threads.
 * No shared memory required.
*/
__global__ void findKbucketsByBlock_kernel (uint * d_bucketCount, uint * d_kVals, uint * d_markedBuckets, uint * d_sums, uint * d_reindexsums, uint * d_bucketBounds, uint * d_KBounds, const int numNewSmallBuckets, const int numOldActive, const int numKs, uint * numUniquePerBlock, uint * markedBucketFlags)
 {
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   int blockKsOffset, blockNumKs;
   if (index < numOldActive) {
     blockKsOffset = d_KBounds[index];
     if (index  + 1 < numOldActive) {
     blockNumKs = d_KBounds[index+1] - blockKsOffset;
     } else {
       blockNumKs = numKs - blockKsOffset;  // potentially unnecessary based on kBounds
     }
     if (blockNumKs > 0){
       numUniquePerBlock[index]=d_findKBucketsByBlock ( d_bucketCount, d_kVals, d_markedBuckets, d_sums, d_reindexsums, numNewSmallBuckets, numNewSmallBuckets*index, d_bucketBounds[index], blockNumKs, blockKsOffset, markedBucketFlags);
     } else {
       numUniquePerBlock[index]=0;
     }
  
   } // end if index
 } // end kernel findKbucketByBlock_kernel

/*
 * This kernel updates the order statistics kVals by block.
 * It should be launched with one block per old active bucket and max threads 
*/
__global__ void updateKValsByBlock (uint * d_kVals, uint * d_sums, uint * d_reindexsums, uint * d_KBounds, const int numKs, const uint numOldActive)
 {
   int threadId = threadIdx.x;
   int blockId = blockIdx.x;
   int index;
   __shared__ uint blockKsOffset;
   __shared__ uint numKsInBlock;

   uint blockKupdate;
   if (blockId<numOldActive) {

     if (threadId<1){
       blockKsOffset = d_KBounds[blockId];
       numKsInBlock = d_KBounds[blockId+1]-blockKsOffset;
     } // end if threadId<1

     syncthreads();

//     if (threadId<numKsInBlock){
     for (index=threadId; index<numKsInBlock; index+=blockDim.x){
       blockKupdate = d_reindexsums[blockKsOffset + threadId] - d_sums[blockKsOffset + threadId];
       d_kVals[blockKsOffset + threadId] += blockKupdate;
     } // end threadId<numKsInBlock

   } // end if (blockId<numOldActive)

 } // end kernel updateKValsByBlock

   



// the host function to find k buckets by block
inline int findKbucketsByBlock (uint * d_bucketCount, uint * d_bucketBounds, uint * d_Kbounds, 
                                 uint * d_reindexsums, uint * d_sums, uint * d_kVals, uint * d_markedBucketFlags, uint * d_numUniquePerBlock, uint *d_uniqueBuckets, uint * numUniqueBuckets, const uint numOldActive, const uint numNewSmallBuckets, const uint numKs)
 {
   uint h_numUnique, h_lastActiveCount, h_newActivePrefix;
   uint *numSelected, *numSelectedSum;
   CUDA_CALL(cudaMalloc(&numSelected, sizeof(uint)));
   CUDA_CALL(cudaMalloc(&numSelectedSum, sizeof(uint)));

   // set threads and compute numBlocks 
   int numFindThreads = 64;
   int numFindBlocks = (int) ceil((float)numOldActive/numFindThreads);
   
   // launch the kernel
   findKbucketsByBlock_kernel<<<numFindBlocks,numFindThreads>>>(d_bucketCount, d_kVals, d_uniqueBuckets, d_sums, d_reindexsums, d_bucketBounds, d_Kbounds, numNewSmallBuckets, numOldActive, numKs,d_numUniquePerBlock, d_markedBucketFlags);

   // get the count of the last active buckeet
   CUDA_CALL(cudaMemcpy(&h_lastActiveCount, d_reindexsums+numKs-1, sizeof(uint), cudaMemcpyDeviceToHost));

   // compute a cumulative sum of reindex sums which currently contains the bucet counts for all marked buckets
   cubDeviceExclusiveSum<uint>(d_reindexsums,d_reindexsums,numKs);

   // update the kVals using the offsets from the old buckets (d_sums) and the new buckets (d_reindexsums)
   updateKValsByBlock<<<numOldActive,MAX_THREADS_PER_BLOCK>>>(d_kVals, d_sums, d_reindexsums, d_Kbounds, numKs, numOldActive);

   // use the flags identifying unique buckets (d_markedBucketFlags) to extra a unique list of active buckets
   // and to extract the appropriate cont offsets for the new data vector 
   SelectFlagged(d_uniqueBuckets,d_uniqueBuckets,d_reindexsums,d_reindexsums,d_markedBucketFlags,numSelected,numKs);
 
   // compute an exlusive sum of the unique buckets per block to obtain a bucket offseet list

   CUDA_CALL(cudaMemcpy(&h_numUnique, numSelected, sizeof(uint), cudaMemcpyDeviceToHost));

   cubDeviceExclusiveSum<uint>(d_numUniquePerBlock,d_numUniquePerBlock,h_numUnique);
 
   // obtain the cummulative count of elements in new active buckets (excluding the last unique bucket)
   CUDA_CALL(cudaMemcpy(&h_newActivePrefix, d_reindexsums+h_numUnique-1, sizeof(uint), cudaMemcpyDeviceToHost));

   // obtain the full length of the new data and the number of unique buckets
   uint newInputLength = h_newActivePrefix + h_lastActiveCount;
   *numUniqueBuckets = h_numUnique;

   //cleanup
   cudaFree(numSelected);
   cudaFree(numSelectedSum);

  return newInputLength;

 } // end findKbucketsByBlock host function



__global__ void ReductionFlags (uint * d_elementToBucket, uint* d_uniqueBuckets, uint* d_blockBounds, uint *d_uniqueBounds, const uint numElements, const uint numOldActive, const uint numNewActive)
 {
   __shared__ uint blockOffset;
   __shared__ uint blockUniqueOffset;
   __shared__ uint numElementsInBlock;
   __shared__ uint numUniqueInBlock;

   extern __shared__ uint activeBuckets[];

   int threadId = threadIdx.x;
   int blockId = blockIdx.x;
   int index;
   uint temp, low, mid, high, compare;
  if (blockId < numOldActive) {
   if (threadId < 1) {
     blockOffset = d_blockBounds[blockId];
     blockUniqueOffset = d_uniqueBounds[blockId];
     if (blockId +1 < numOldActive) {
       numElementsInBlock = d_blockBounds[blockId+1]-blockOffset;
       numUniqueInBlock = d_uniqueBounds[blockId+1]-blockUniqueOffset;
     } else {
       numElementsInBlock = numElements-blockOffset;
       numUniqueInBlock = numNewActive-blockUniqueOffset;
     }
   }

   syncthreads();

   // read unique active buckets into shared memory
   for (int i = threadId; i < numUniqueInBlock; i += blockDim.x) {
     activeBuckets[i] = d_uniqueBuckets[blockUniqueOffset + i];
   }
   activeBuckets[numUniqueInBlock]=2200000; // ensure the binary search does not incorrectly identify a bucket as active


   syncthreads();

   // binary search the active buckets for this block
   for (int i = threadId; i < numElementsInBlock; i += blockDim.x) {
     index = blockOffset + i;
     temp = d_elementToBucket[index];
     low = 0;
     high = numUniqueInBlock;
     compare = 0;  
     for (int j = 1; j < numUniqueInBlock + 1; j *= 2) {
       mid = (high + low) / 2;
       compare = (temp > activeBuckets[mid]);
       low = compare ? mid : low;
       high = compare ? high : mid;
     } //end for

     // if the current bucket is active, flag as 1 otherwise flag as 0 using elementToBucket as flag vector
     if (temp==activeBuckets[high]) {
       d_elementToBucket[index]=1;
     } else {
       d_elementToBucket[index]=0;
     } // end if else temp==active

    } // end if threadID< numElementsInBlock
  } // end if blockId<numOldActive
 } // end kernel ReductionFlags
    
  
template<typename T>
int Reduction(T* d_vec, T* d_new, uint * d_elementToBucket, uint* d_uniqueBuckets, uint* d_blockBounds, uint *d_uniqueBounds, const uint numElements, const uint numOldActive, const uint numNewActive)
 {  
    uint *numSelected;
    uint h_numSelected;
    CUDA_CALL(cudaMalloc(&numSelected, sizeof(uint)));

    // identify the active buckets by block and mark them with flags in d_elementToBucket  
    ReductionFlags<<<numOldActive,MAX_THREADS_PER_BLOCK,numOldActive*sizeof(uint)>>>(d_elementToBucket, d_uniqueBuckets, d_blockBounds, d_uniqueBounds, numElements, numOldActive, numNewActive);

    // use the flags from the previous kernel select the elements from d_vec
    cubDeviceSelectFlagged<T>(d_vec, d_elementToBucket, d_new, numSelected, numElements);

    CUDA_CALL(cudaMemcpy(&h_numSelected, numSelected, sizeof(uint), cudaMemcpyDeviceToHost));

    cudaFree(numSelected);

    return h_numSelected;
 }  // end function Reduction








