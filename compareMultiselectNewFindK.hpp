#ifndef COMPAREMULTISELECT_H_
#define COMPAREMULTISELECT_H_

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>

#include <algorithm>
//Include various thrust items that are used
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>

#include "cubdevicesort.cu"
#include "mgpudevicesort.cu"
//#include "quickMultiSelect.cpp"

//various functions, include the functions
//that print numbers in binary.
#include "printFunctions.cu"

//the algorithms
#include "bucketMultiselect.cu"
//#include "bucketMultiselectNew2.cu"
#include "bucketMultiselectNewFindK.cu"
#include "bucketMultiselect_thrust.cu"
#include "bucketMultiselect_cub.cu"
#include "bucketMultiselect_mgpu.cu"
//#include "naiveBucketMultiselect.cu"

#include "generateProblems.cu"
#include "multiselectTimingFunctionsNewFindK.cu"

#define NUMBEROFALGORITHMS 3
char* namesOfMultiselectTimingFunctions[NUMBEROFALGORITHMS] = 
  {"Sort and Choose Multiselect", "Bucket Multiselect", "Bucket MultiselectNewFindK"}; //, "Original Bucket Multiselect"};

namespace CompareMultiselectNewFindK {

  using namespace std;

  template<typename T>
  void compareMultiselectAlgorithms(uint size, uint* kVals, uint numKs, uint numTests
                                    , uint *algorithmsToTest, uint generateType, uint kGenerateType
                                    , char* fileNamecsv, int world_rank, int world_size, char* processor_name
                                    , uint datatype, T* data = NULL) {

    // allocate space for operations
    T *h_vec, *h_vec_copy;
    T *** resultsArray = (T***) malloc (NUMBEROFALGORITHMS * sizeof (T**));
    float** timeArray = (float**) malloc (NUMBEROFALGORITHMS * sizeof (float*));
    for (int i = 0; i < NUMBEROFALGORITHMS; i++) {
      timeArray[i] = (float*) malloc (numTests * sizeof(float));
      resultsArray[i] = (T**) malloc (numTests * sizeof(float));
    }
    float totalTimesPerAlgorithm[NUMBEROFALGORITHMS];
    uint* winnerArray = (uint *) malloc (numTests * sizeof (uint));
    uint* timesWon = (uint*) malloc (NUMBEROFALGORITHMS * sizeof (uint));
    int i,j,m,x;
    int runOrder[NUMBEROFALGORITHMS];
    unsigned long long seed;
    results_t<T> *temp;
    ofstream fileCsv;
    timeval t1;
 
    typedef results_t<T>* (*ptrToTimingFunction)(T*, uint, uint *, uint, int, int, char*, uint, uint);
    typedef void (*ptrToGeneratingFunction)(T*, uint, curandGenerator_t);

    //these are the functions that can be called
    ptrToTimingFunction arrayOfTimingFunctions[NUMBEROFALGORITHMS] = 
      {&timeSortAndChooseMultiselect_a<T>,
       &timeBucketMultiselect_a<T>,
       &timeBucketMultiselectNewFindK_a<T>};
    //, &timeBucketMultiselect_thrust<T>};
  
    ptrToGeneratingFunction *arrayOfGenerators;
    char** namesOfGeneratingFunctions;
  
    // this is the array of names of functions that generate problems of this type, 
    // ie float, double, or uint
    namesOfGeneratingFunctions = returnNamesOfGenerators<T>();
    arrayOfGenerators = (ptrToGeneratingFunction *) returnGenFunctions<T>();

    if (world_rank == 0) 
      printf("Files will be written to %s\n", fileNamecsv);
    fileCsv.open(fileNamecsv, ios_base::app);
    
    //zero out the totals and times won
    bzero(totalTimesPerAlgorithm, NUMBEROFALGORITHMS * sizeof(uint));
    bzero(timesWon, NUMBEROFALGORITHMS * sizeof(uint));

    //allocate space for h_vec, and h_vec_copy
    if (world_rank == 0) {
      h_vec = (T *) malloc(size * sizeof(T));
      h_vec_copy = (T *) malloc(size * sizeof(T));
    }

    //create the random generator.
    curandGenerator_t generator;
    srand(unsigned(time(NULL)));

    if (world_rank == 0) {
      printf("The distribution is: %s\n", namesOfGeneratingFunctions[generateType]);
      printf("The k distribution is: %s\n", namesOfKGenerators[kGenerateType]);
    }
    
    /***********************************************/
    /*********** START RUNNING TESTS ************
  /***********************************************/


    for(i = 0; i < numTests; i++) {
      gettimeofday(&t1, NULL);
      seed = t1.tv_usec * t1.tv_sec;
      if (world_rank == 0) {
        for(m = 0; m < NUMBEROFALGORITHMS;m++)
          runOrder[m] = m;
    
        std::random_shuffle(runOrder, runOrder + NUMBEROFALGORITHMS);
        // Send out random run order so all processes run together.
        for (int jj = 1; jj < world_size; jj++)
          MPI_Send(runOrder, NUMBEROFALGORITHMS, MPI_INT, jj, 0, MPI_COMM_WORLD);
      }
      // Receive random run order
      else 
        MPI_Recv(runOrder, NUMBEROFALGORITHMS, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      fileCsv << size << "," << numKs << "," << 
        namesOfGeneratingFunctions[generateType] << "," << 
        namesOfKGenerators[kGenerateType] << ",";

      curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetPseudoRandomGeneratorSeed(generator,seed);
      if (world_rank == 0) {
        printf("Running test %u of %u for size: %u and numK: %u\n", i + 1, 
               numTests, size, numKs);
      }

      // Use the root process to randomly generate the data set
      if (world_rank == 0) {
        //generate the random vector using the specified distribution
        if(data == NULL) 
          arrayOfGenerators[generateType](h_vec, size, generator);
        else
          h_vec = data;

        //copy the vector to h_vec_copy, which will be used to restore it later
        memcpy(h_vec_copy, h_vec, size * sizeof(T));
      }

      //Scatter h_vec 
      int* displs = (int*) malloc (world_size * sizeof(int));
      int* sendcounts = (int*) malloc (world_size * sizeof(int));
      int offset = (int) size / world_size;
      for (int ii = 0; ii < world_size - 1; ii++) {
        sendcounts[ii] = offset;
        displs[ii] = ii * offset;
      }
      sendcounts[world_size - 1] = size - (world_size - 1) * offset;
      displs[world_size - 1] = (world_size - 1) * offset;
      int newSize = sendcounts[world_rank];
      // allocate h_vecChunk to receive the chunk of h_vec
      T* h_vecChunk = (T*) malloc (newSize * sizeof(T));
      int recv = sendcounts[world_rank];
      MPI_Barrier(MPI_COMM_WORLD);
      switch (datatype) {
      case 0:
        MPI_Scatterv (h_vec, sendcounts, displs, MPI_FLOAT, h_vecChunk, recv, MPI_FLOAT, 0, MPI_COMM_WORLD);  
        break;
      case 1:
        MPI_Scatterv (h_vec, sendcounts, displs, MPI_DOUBLE, h_vecChunk, recv, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
        break;
      case 2:
        MPI_Scatterv (h_vec, sendcounts, displs, MPI_UNSIGNED, h_vecChunk, recv, MPI_UNSIGNED, 0, MPI_COMM_WORLD);  
        break;
      }

      winnerArray[i] = 0;
      float currentWinningTime = INFINITY;
      //run the various timing functions
      for(x = 0; x < NUMBEROFALGORITHMS; x++){
        j = runOrder[x];
        if(algorithmsToTest[j]){

          //run timing function j. If it is the distributed multiselect, use different arguments.
          if (j == 2)
            temp = arrayOfTimingFunctions[j](h_vecChunk, newSize, kVals, numKs, world_rank, world_size, processor_name, datatype, size);
          else if (world_rank == 0) {
            // Must take size as a parameter twice to fit the template
            temp = arrayOfTimingFunctions[j](h_vec_copy, size, kVals, numKs, world_rank, world_size, processor_name, datatype, size);
          }

          if (world_rank == 0) {
            //record the time result
            timeArray[j][i] = temp->time;
            //record the value returned
            resultsArray[j][i] = temp->vals;
            //update the current "winner" if necessary
            if(timeArray[j][i] < currentWinningTime){
              currentWinningTime = temp->time;
              winnerArray[i] = j;
            }
          }
          //perform clean up 
          if (world_rank == 0 && x != 2) {
            //free (temp);
            memcpy(h_vec_copy, h_vec, size * sizeof(T));
          }
        }
      }

      curandDestroyGenerator(generator);
      if (world_rank == 0) {
        for(x = 0; x < NUMBEROFALGORITHMS; x++)
          if(algorithmsToTest[x])
            fileCsv << namesOfMultiselectTimingFunctions[x] << "," << timeArray[x][i] << ",";
      }

      if (world_rank == 0) {
        // check for errors, and output information to recreate problem
        uint flag = 0;
        for(m = 1; m < NUMBEROFALGORITHMS;m++)
          if(algorithmsToTest[m])
            for (j = 0; j < numKs; j++) {
              if(resultsArray[m][i][j] != resultsArray[0][i][j]) {
                flag++;
                fileCsv << "\nERROR ON TEST " << i << " of " << numTests << " tests!!!!!\n";
                fileCsv << "vector size = " << size << "\nvector seed = " << seed << "\n";
                fileCsv << "numKs = " << numKs << "\n";
                fileCsv << "wrong k = " << kVals[j] << " kIndex = " << j << 
                  " wrong result = " << resultsArray[m][i][j] << " correct result = " <<  
                  resultsArray[0][i][j] << "\n";
                std::cout <<namesOfMultiselectTimingFunctions[m] <<
                  " did not return the correct answer on test " << i + 1 << " at k[" << j << 
                  "].  It got "<< resultsArray[m][i][j];
                std::cout << " instead of " << resultsArray[0][i][j] << ".\n" ;
                std::cout << "RESULT:\t";
                PrintFunctions::printBinary(resultsArray[m][i][j]);
                std::cout << "Right:\t";
                PrintFunctions::printBinary(resultsArray[0][i][j]);
              }
            }

        fileCsv << flag << "\n";
      }
    } // end for
    if (world_rank == 0) {
      //calculate the total time each algorithm took
      for(i = 0; i < numTests; i++)
        for(j = 0; j < NUMBEROFALGORITHMS;j++)
          if(algorithmsToTest[j])
            totalTimesPerAlgorithm[j] += timeArray[j][i];

      //count the number of times each algorithm won. 
      for(i = 0; i < numTests; i++) {
        timesWon[winnerArray[i]]++;
      }
      printf("\n\n");
        
      //print out the average times

      for(i = 0; i < NUMBEROFALGORITHMS; i++)
        if(algorithmsToTest[i])
          printf("%-20s averaged: %f ms\n", namesOfMultiselectTimingFunctions[i], totalTimesPerAlgorithm[i] / numTests);

      for(i = 0; i < NUMBEROFALGORITHMS; i++)
        if(algorithmsToTest[i])
          printf("%s won %u times\n", namesOfMultiselectTimingFunctions[i], timesWon[i]);
    } // end if (world_rank == 0)

    // free results
    if ((world_rank == 0 && x != 2) || (x == 2)) {
      for(i = 0; i < numTests; i++) 
        for(m = 0; m < NUMBEROFALGORITHMS; m++) 
          if(algorithmsToTest[m])
            free(resultsArray[m][i]);
    }
    //free h_vec and h_vec_copy
    if ((world_rank == 0 && x != 2) || (x == 2)) {
      if(data == NULL) 
        free(h_vec);
      free(h_vec_copy);
    }

    //close the file
    fileCsv.close();
  }


  /* This function generates the array of kVals to work on and acts as a wrapper for 
     comparison.
  */
  template<typename T>
  void runTests (uint generateType, char* fileName, uint startPower, uint stopPower
                 , uint timesToTestEachK, uint kDistribution, uint startK, uint stopK, uint kJump
                 , int world_rank, int world_size, char* processor_name, uint type) {
    uint algorithmsToRun[NUMBEROFALGORITHMS]= {1, 1, 1};
    uint size;
    uint i;
    uint arrayOfKs[stopK+1];
  
    // double the array size to the next powers of 2
    for(size = (1 << startPower); size <= (1 << stopPower); size *= 2) {
      if (world_rank == 0) {
        unsigned long long seed;
        timeval t1;
        gettimeofday(&t1, NULL);
        seed = t1.tv_usec * t1.tv_sec;
        curandGenerator_t generator;
        srand(unsigned(time(NULL)));
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(generator,seed);

        arrayOfKDistributionGenerators[kDistribution](arrayOfKs, stopK, size, generator);

        curandDestroyGenerator(generator);

        // Send k values to all processors
        for (int j = 1; j < world_size; j++)
          MPI_Send(arrayOfKs, stopK + 1, MPI_UNSIGNED, j, 0, MPI_COMM_WORLD);
      }

      // Receive k values
      else {
        MPI_Recv(arrayOfKs, stopK + 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      for(i = startK; i <= stopK; i+=kJump) {
        cudaDeviceReset();
        cudaThreadExit();
        if (world_rank == 0)
          printf("NOW ADDING ANOTHER K\n\n");
        compareMultiselectAlgorithms<T>(size, arrayOfKs, i, timesToTestEachK, 
                                        algorithmsToRun, generateType, kDistribution, fileName
                                        , world_rank, world_size, processor_name, type);
      }
    }
  }
}

#endif
