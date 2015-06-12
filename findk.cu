/*
 * This device function will be called by a kernel with each thread finding the 
 * k buckets for each oldActiveBucket (i.e. each block used to reassign buckets) 
*/
__device__ void d_findKBucketsByBlock(uint * d_bucketCount, uint * kVals, uint * markedBuckets, 
                                    uint * sums, const int numNewSmallBuckets, 
                                    const int blockBucketOffset, const int blockStart,  
                                    const int bockNumKs, const int blockKsOffset)
 {
    int kBucket = blockBucketOffset;
    int blockMaxBucket = blockBucketOffset + numNewSmallBuckets;
    int k;
    int sum = blockStart;
    uint temp;

    for(register int i = 0; i < blockNumKs; i++) {
      k = kVals[blockKsOffset + i];
      while ((sum < k) & (kBucket < blockMaxBucket)) {
        temp = d_bucketCount[kBucket];
        sum += temp;     
        kBucket++;
      } // end while
      markedBuckets[blockKsOffset + i] = kBucket;

      sums[blockKsOffset + i] = sum - temp; 

    } // end for

 } // end device function d_findKBucketsByBlock

/*
 * The kernel will launch one thread per old active bucket to get the new k buckets
 * which should be identified as active.  It also gets the sum of all elements that 
 * were in previous new buckets in order to update the desired order statistics.
 * Launch this kernel with a fixed number of threads per block, probably 64, and enough 
 * blocks to achieve numOldActive total threads.
 * No shared memory required.
*/
__global__ void findKbucketsByBlock_kernel (uint * d_bucketCount, uint * d_kVals, uint * d_markedBuckets, uint * d_sums, uint * d_bucketBounds, uint * d_KBounds, const int numNewSmallBuckets, const int numOldActive)
 {
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   int blockKsOffset, blockNumKs;
   if (index < numOldActive) {
     blockKsOffset = d_KBounds[index];
     blockNumKs = d_KBounds[index+1] - blockKsOffset;
     d_findKBucketsByBlock ( d_bucketCount, d_kVals, d_markedBuckets, d_sums,
                             numNewSmallBuckets, numNewSmallBuckets*index, d_bucketBounds[index],
                             blockNumKs, blockKsOffset);
   } // end if index
 } // end kernel findKbucketByBlock_kernel



// the host function to find k buckets by block
inline void findKbucketsByBlock ()
 {

// create device variables for marked buckets, sums, kVals
   uint *d_kVals, *d_markedBuckets, *d_sums;
   CUDA_CALL(cudaMalloc(&d_kVals, numKs * sizeof(uint)));
   CUDA_CALL(cudaMalloc(&d_markedBuckets, numKs * sizeof(uint)));
   CUDA_CALL(cudaMalloc(&d_sums, numKs * sizeof(uint)));

   uint *d_blockKsOffset;
   CUDA_CALL(cudaMalloc(&d_blockKsOffset, numOldActive * sizeof(uint)));
// (since this is potentiall recursive, these should be preallocated)

// compute Kbounds and copy to device
   Kbounds[0]=0;
   uint j = 0;
   uint i = 1;
   uint k = kVals[j];
   while (i < numOldActive) {
     blockOffset = h_blockBounds[i];
     while ( (k < blockOffset) && (j < numKs) ) {
       j++;
       k = kVals[j];
     }
     Kbounds[i]=j-1;
     i++;
   }
   Kbounds[numOldActive]=numKs-1;

   CUDA_CALL(cudaMemcpy(d_Kbounds, Kbounds, 
                         (numOldActive+1) * sizeof(uint), cudaMemcpyHostToDevice));


   // set threads and compute numBlocks 
   int numFindThreads = 64;
   int numFindBuckets = (int) ceil((float)numOldActive/numFindThreads);
   
   // launch the kernel
   findKbucketsByBlock_kernel<<<numFindBuckets,numFindThreads>>>(d_bucketCount, d_kVals, d_markedBuckets, d_sums, d_bucketBounds, d_Kbounds, numNewSmallBuckets, numOldActive);

   //copy markedbuckets and sums back to host
   CUDA_CALL(cudaMemcpy(markedBuckets, d_markedBuckets, 
                         numKs * sizeof(uint), cudaMemcpyDeviceToHost));
   CUDA_CALL(cudaMemcpy(KthBucketScanner, d_sums, 
                         numKs * sizeof(uint), cudaMemcpyDeviceToHost));

 } // end findKbucketsByBlock host function

