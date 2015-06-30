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

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>

// include compareMultiselectNew2 library
#include "compareMultiselectNewFindK.hpp"

/* main fucntion that takes user input to run compare Multiselect on
 */

int main (int argc, char *argv[]) {

  MPI_Init (NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  char *fileName, *hostName, *typeString;

  fileName = (char*) malloc(128 * sizeof(char));
  typeString = (char*) malloc(10 * sizeof(char));
  hostName = (char*) malloc(20 * sizeof(char));
  gethostname(hostName, 20);

  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  char * humanTime = asctime(timeinfo);
  humanTime[strlen(humanTime)-1] = '\0';

  uint testCount, type,distributionType,startPower,stopPower,kDistribution,startK
    ,stopK,jumpK;
  
  // On root process, collect user input
  if (world_rank == 0) {
    printf("Please enter the type of value you want to test:\n0-float\n1-double\n2-uint\n");
    scanf("%u", &type);
    printf("Please enter distribution type: ");
    printDistributionOptions(type);
    scanf("%u", &distributionType);
    printf("Please enter K distribution type: ");
    printKDistributionOptions();
    fflush(stdout);
    scanf("%u", &kDistribution);
    printf("Please enter Start power: ");
    fflush(stdout);
    scanf("%u", &startPower);
    printf("Please enter Stop power: ");
    fflush(stdout);
    scanf("%u", &stopPower); 
    printf("Please enter Start number of K values: ");
    fflush(stdout);
    scanf("%u", &startK);
    printf("Please enter number of K values to jump by: ");
    fflush(stdout);
    scanf("%u", &jumpK);
    printf("Please enter Stop number of K values: ");
    fflush(stdout);
    scanf("%u", &stopK);
    printf("Please enter number of tests to run per K: ");
    fflush(stdout);
    scanf("%u", &testCount);

    switch(type){
    case 0:
      typeString = "float";
      break;
    case 1:
      typeString = "double";
      break;
    case 2:
      typeString = "uint";
      break;
    default:
      break;
    }

    snprintf(fileName, 128, 
             "%s %s k-dist:%s 2^%d to 2^%d (%d:%d:%d) %d-tests on %s at %s", 
             typeString, getDistributionOptions(type, distributionType), 
             getKDistributionOptions(kDistribution), startPower, stopPower, 
             startK, jumpK, stopK, testCount, hostName, humanTime);
    printf("File Name: %s \n", fileName);
  
    // Send all data from user to all other MPI process
    for (int i = 1; i < world_size; i++) {
      MPI_Send(&type, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      MPI_Send(&distributionType, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      MPI_Send(fileName, 128, MPI_CHAR, i, 0, MPI_COMM_WORLD);
      MPI_Send(&startPower, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      MPI_Send(&stopPower, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD); 
      MPI_Send(&testCount, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      MPI_Send(&kDistribution, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      MPI_Send(&startK, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      MPI_Send(&stopK, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      MPI_Send(&jumpK, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
    }
  }

  else {
    MPI_Recv(&type, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&distributionType, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(fileName, 128, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&startPower, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&stopPower, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    MPI_Recv(&testCount, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&kDistribution, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&startK, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&stopK, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&jumpK, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  using namespace CompareMultiselectNewFindK;

  MPI_Barrier(MPI_COMM_WORLD);

  switch(type){
  case 0:
    runTests<float>(distributionType,fileName,startPower,stopPower,testCount,
                    kDistribution,startK,stopK,jumpK,world_rank,world_size,processor_name);
    break;
  case 1:
    runTests<double>(distributionType,fileName,startPower,stopPower,testCount,
                     kDistribution,startK,stopK,jumpK,world_rank,world_size,processor_name);
    break;
  case 2:
    runTests<uint>(distributionType,fileName,startPower,stopPower,testCount,
                   kDistribution,startK,stopK,jumpK,world_rank,world_size,processor_name);
    break;
  default:
    if (world_rank == 0)
      printf("You entered and invalid option, now exiting\n");
    break;
  }


  //printf ("On host %d named %s, type is %u\n", world_rank, processor_name, type);  

  free (fileName);

  MPI_Finalize();

  return 0;
}

