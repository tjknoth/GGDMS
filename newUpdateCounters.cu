inline int newUpdateCounters(uint * uniqueBuckets, uint * kthBuckets, uint * reindexCounter,  uint * kVals, uint * kthBucketScanner,  uint * h_bucketCount, uint * d_numActivePerBlock, int numKs, uint numNewSmallBuckets, uint newNumActive)
{

    // we must update K since we have reduced the problem size to elements in the 
    // kth bucket.
    //  get the index of the first element
    //  add the number of elements

    int UniqueCounter;
    uint currentMarkedBucket, lastMarkedBucket;
    uint numBlocksPerBucket[newNumActive];

    uniqueBuckets[0] = kthBuckets[0];
    reindexCounter[0] = 0;
    UniqueCounter = 1;
    kVals[0] -= kthBucketScanner[0];
    uint blocksBucketOffset = numNewSmallBuckets;
    int j = 0;
    numBlocksPerBucket[0]=0;
    currentMarkedBucket = kthBuckets[0];

    // the first marked bucket must be in the first old active bucket
    //numBlocksPerBucket[j]+=1;

    for (int i = 1; i < numKs; i++) {
      lastMarkedBucket = currentMarkedBucket;
      currentMarkedBucket = kthBuckets[i];
      
      if (currentMarkedBucket != lastMarkedBucket) {
        uniqueBuckets[UniqueCounter] = currentMarkedBucket;
        reindexCounter[UniqueCounter] = 
          reindexCounter[UniqueCounter-1]  + h_bucketCount[lastMarkedBucket];
        UniqueCounter++;
      } // end if (currentMarkedBucket
      kVals[i] = reindexCounter[UniqueCounter-1] + kVals[i] - kthBucketScanner[i];
    }  // end for i<numKs

    for (int i = 0; i < UniqueCounter; i++) {
      currentMarkedBucket = uniqueBuckets[i];
        if (currentMarkedBucket<blocksBucketOffset) {
printf("\n\n yes \n\n");
          numBlocksPerBucket[j]+=1;
        } else {
            j++;
            blocksBucketOffset+=numNewSmallBuckets;
            numBlocksPerBucket[j]=0;
            i--;
        } // end if - else currentMarkedBucket

    } // end for i<uniqueCounter

      for (int i=0; i< numKs; i+=3) printf ("kvals[%d] = %d\t kvals[%d] = %d\t kvals[%d] = %d\n", i, kVals[i],i+1, kVals[i+1],i+2, kVals[i+2]);
      for (int j=0; j<newNumActive; j+=3) printf ("numBlocksPerBucket[%d]=%d\t numBlocksPerBucket[%d]=%d\t numBlocksPerBucket[%d]=%d\n", j,numBlocksPerBucket[j],j+1,numBlocksPerBucket[j+1],j+2,numBlocksPerBucket[j+2]);
    


    CUDA_CALL(cudaMemcpy(d_numActivePerBlock, numBlocksPerBucket, 
                         newNumActive * sizeof(uint), cudaMemcpyHostToDevice));

    return UniqueCounter;


 } // end function
    

/*
newUpdateCounters(  uniqueBuckets, kthBuckets, reindexCounter,  kVals, kthBucketScanner,  h_bucketCount, d_numActivePerBlock, numUniqueBuckets, numKs, numNewSmallBuckets, newNumActive)
*/


