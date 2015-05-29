
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sys/time.h>
#include <unistd.h>


#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#define CUDART_PI_F 3.141592654f

//Include various thrust items that are used
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <thrust/sort.h>

#include "cubdevicesort.cu"
#include "mgpudevicesort.cu"

//#include "moderngpu/include/moderngpu.cuh"
//#include "moderngpu/include/mgpudevice.cuh"
//#include "moderngpu/include/kernels/mergesort.cuh"



using namespace std;

template<typename T>
void generateUniformFloats(T *h_vec, uint numElements, curandGenerator_t generator){
  float * d_generated;
  cudaMalloc(&d_generated, numElements * sizeof(T));
  curandGenerateUniform(generator, d_generated,numElements);
  cudaMemcpy(h_vec, d_generated, numElements * sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_generated);
}

template <typename T>
void createProblem(T *h_vec, uint numElements)
{
  
  curandGenerator_t generator;
  unsigned long long seed; 
  timeval t1;

  gettimeofday(&t1, NULL);
  seed = t1.tv_usec * t1.tv_sec;

  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator,seed);

  generateUniformFloats<T>(h_vec, numElements, generator);

  curandDestroyGenerator(generator);
}




/*
******************************
** Main Function which creates a problem and sorts it with all the algorithms, then compares them to make sure they are the same **
******************************
*/
template <typename T> 
void compareSorts(uint numElements, float *timings)        //, mgpu::CudaContext& context)
{
  // create the variables on the host
  T *h_thrust, *h_cub, *h_mgpu, *h_copy;
  h_thrust = (T *) malloc(numElements * sizeof(T));
  h_cub = (T *) malloc(numElements * sizeof(T));
  h_mgpu = (T *) malloc(numElements * sizeof(T));
  h_copy = (T *) malloc(numElements * sizeof(T));

  // create the device vector
  T *d_vec;
  cudaMalloc(&d_vec, numElements*sizeof(T));

  // create the timing variables
  float time_thrust, time_cub, time_mgpu;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // random order shuffler
  int runOrder[3];
  runOrder[0]=0;
  runOrder[1]=1;
  runOrder[2]=2;
  random_shuffle(runOrder, runOrder+3);

  // create a random problem
  createProblem<T>(h_copy, numElements);
 
         thrust::device_ptr<T> dev_ptr(d_vec);
         // warm up the gpu
         thrust::sort(dev_ptr, dev_ptr + numElements);


//  mgpu::ContextPtr context = mgpu::CreateCudaDevice(1, 0, false);

  for (int j=0; j<3; j++){
    switch ( runOrder[j] ){
      case 0 :
        // ***********  MGPU ******************

        // start timer for mgpu
       cudaEventRecord(start, 0);

        // copy to device and mgpu sort
        cudaMemcpy(d_vec, h_copy, numElements * sizeof(T), cudaMemcpyHostToDevice);

        //  mgpuDeviceSort(d_vec, numElements, context);
//        mgpu::MergesortKeys(d_vec, numElements, *context);
        mgpuDeviceSort<T>(d_vec, numElements);
        cudaMemcpy(h_mgpu, d_vec, numElements*sizeof(T), cudaMemcpyDeviceToHost);

        // stop timer and record elapsed time
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_mgpu, start, stop);
        timings[2] += time_mgpu;
       break;

       case 1 :
         // ***********  THRUST ******************

         // start timer for thrust
         cudaEventRecord(start, 0);


         // copy to device and thrust sort
         cudaMemcpy(d_vec, h_copy, numElements * sizeof(T), cudaMemcpyHostToDevice);

//         thrust::device_ptr<float> dev_ptr(d_vec);
         thrust::sort(dev_ptr, dev_ptr + numElements);
         cudaMemcpy(h_thrust, d_vec, numElements*sizeof(T), cudaMemcpyDeviceToHost);

         // stop timer and record elapsed time
         cudaEventRecord(stop, 0);
         cudaEventSynchronize(stop);
         cudaEventElapsedTime(&time_thrust, start, stop);
         timings[0] += time_thrust;
       break;

       case 2 :
         // ***********  CUB  ******************
  
         // start timer for cub
         cudaEventRecord(start, 0);

         // copy to device and cub sort
         cudaMemcpy(d_vec, h_copy, numElements * sizeof(T), cudaMemcpyHostToDevice);

         cubDeviceSort<T>(d_vec, numElements);
         cudaMemcpy(h_cub, d_vec, numElements*sizeof(T), cudaMemcpyDeviceToHost);

         // stop timer and record elapsed time
         cudaEventRecord(stop, 0);
         cudaEventSynchronize(stop);
         cudaEventElapsedTime(&time_cub, start, stop);
         timings[1] += time_cub;
       break;

       default:
         cout << "Something went wrong with the switch statement." << endl;
    } // ends the switch
  } // ends the for loop

/*
  int mid=0, far=0;
  for (int i=0; i<15; i++){
    mid=numElements/2+i;
    far = numElements-i-1;
    cout << "h_t[" << i << "]= " << h_thrust[i] << "  ";
    cout << "h_c[" << i << "]= " << h_cub[i] << "  ";
    cout << "h_m[" << i << "]= " << h_mgpu[i] << "      ";
    cout << "h_t[" << mid << "]= " << h_thrust[mid] << "  ";
    cout << "h_c[" << mid << "]= " << h_cub[mid] << "  ";
    cout << "h_m[" << mid << "]= " << h_mgpu[mid] << "      ";
    cout << "h_t[" << far << "]= " << h_thrust[far] << "  ";
    cout << "h_c[" << far << "]= " << h_cub[far] << "  ";
    cout << "h_m[" << far << "]= " << h_mgpu[far] << "  ";
    cout << endl;
  }
*/

//  cout << "Timings for numElements = " << numElements << " are:" << endl << "  Thrust: " << time_thrust << endl << "     Cub: " <<  time_cub << endl << "    MGPU: " << time_mgpu << endl;
  free(h_thrust);
  free(h_cub);
  free(h_mgpu);
  free(h_copy);
  cudaFree(d_vec);

}


int main (int argc, char** argv) {

//  mgpu::ContextPtr context = mgpu::CreateCudaDevice(argc, argv, false);

  uint type, startPower, stopPower, testCount;
  uint numElements, testnumber; 
  float timings[3];
  timings[0]=0.0f;
  timings[1]=0.0f;
  timings[2]=0.0f;
  
  printf("Please enter a type (0 = int, 1 = float, 2 = double): ");
  scanf("%u", &type);
  printf("Please enter Start power: ");
  scanf("%u", &startPower);
  printf("Please enter Stop power: ");
  scanf("%u", &stopPower); 
  printf("Please enter number of tests to run per K: ");
  scanf("%u", &testCount);

  float testcountinverse = 1/((float) testCount);

  switch (type) {
   case 0: 
    for (numElements = (1 << startPower); numElements <= (1 << stopPower); numElements *= 2){
      for (testnumber = 0; testnumber < testCount; testnumber++){
        compareSorts<int>(numElements, timings);  //, *context);
      }
      timings[0] *= testcountinverse;
      timings[2] *= testcountinverse;
      timings[1] *= testcountinverse;
      cout << "Type: INT. Average Timings for " << testCount << " tests with numElements = " << numElements << " are:" << endl << "  Thrust: " << timings[0] << endl << "     Cub: " <<  timings[1] << endl << "    MGPU: " << timings[2] << endl;
    }
   break;
   case 1: 
    for (numElements = (1 << startPower); numElements <= (1 << stopPower); numElements *= 2){
      for (testnumber = 0; testnumber < testCount; testnumber++){
        compareSorts<float>(numElements, timings);  //, *context);
      }
      timings[0] *= testcountinverse;
      timings[2] *= testcountinverse;
      timings[1] *= testcountinverse;
      cout << "Type: FLOAT. Average Timings for " << testCount << " tests with numElements = " << numElements << " are:" << endl << "  Thrust: " << timings[0] << endl << "     Cub: " <<  timings[1] << endl << "    MGPU: " << timings[2] << endl;
    }
   break;
   case 2: 
    for (numElements = (1 << startPower); numElements <= (1 << stopPower); numElements *= 2){
      for (testnumber = 0; testnumber < testCount; testnumber++){
        compareSorts<double>(numElements, timings);  //, *context);
      }
      timings[0] *= testcountinverse;
      timings[2] *= testcountinverse;
      timings[1] *= testcountinverse;
      cout << "Type: DOUBLE. Average Timings for " << testCount << " tests with numElements = " << numElements << " are:" << endl << "  Thrust: " << timings[0] << endl << "     Cub: " <<  timings[1] << endl << "    MGPU: " << timings[2] << endl;
    }
   break;
   default: cout << "Invalid response to 'Please enter a type: 0 = int, 1 = float, 2 = double' " << endl;
  } // end switch

  return 0;
}


