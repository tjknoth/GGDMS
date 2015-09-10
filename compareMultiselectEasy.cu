// -*-c++-*-

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

// include compareMultiselectNew2 library
#include "compareMultiselectNewFindK.hpp"

/* main fucntion that takes user input to run compare Multiselect on
 */

void printChoices (uint* choice) {
  printf("0. Uniform distribution, uniform Ks, start power = 24, stop power = 24, 101 Ks\n");
  printf("1. Uniform distribution, uniform random Ks, start power = 27, stop power = 27, 1001 Ks\n");
  printf("2. Uniform distribution, uniform random Ks, start power = 25, stop power = 25, 101 Ks\n");
  printf("3. Uniform distribution, uniform Ks, start power = 15, stop power = 15, 101 Ks\n");
  printf("4. Custom test\n");
  scanf("%u", choice);
}

int main (int argc, char *argv[]) {

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

  uint testCount,type,distributionType,startPower,stopPower,kDistribution,startK
    ,stopK,jumpK,testChoice;
  
  printf("Please enter the type of value you want to test:\n0-float\n1-double\n2-uint\n");
  scanf("%u", &type);
  printf("Please select the number of tests to run per K: ");
  scanf("%u", &testCount);
  printf("Please select the other parameters: \n");
  printChoices(&testChoice);
  if (testChoice == 4) {
    printf("Please enter distribution type: ");
    printDistributionOptions(type);
    scanf("%u", &distributionType);
    printf("Please enter K distribution type: ");
    printKDistributionOptions();
    scanf("%u", &kDistribution);
    printf("Please enter Start power: ");
    scanf("%u", &startPower);
    printf("Please enter Stop power: ");
    scanf("%u", &stopPower); 
    printf("Please enter Start number of K values: ");
    scanf("%u", &startK);
    printf("Please enter number of K values to jump by: ");
    scanf("%u", &jumpK);
    printf("Please enter Stop number of K values: ");
    scanf("%u", &stopK);
    printf("Please enter number of tests to run per K: ");
    scanf("%u", &testCount);
  }

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
  printf ("testChoice = %u\n", testChoice);


  using namespace CompareMultiselectNew2;
  
  
  switch(testChoice) {
  case 0:
    distributionType = 0;
    kDistribution = 1;
    startPower = 24;
    stopPower = 24;
    startK = 101;
    jumpK = 1;
    stopK = 101;
    break;
  case 1:
    distributionType = 0;
    kDistribution = 0;
    startPower = 27;
    stopPower = 27;
    startK = 1001;
    jumpK = 1;
    stopK = 1001;
    break;
  case 2:
    distributionType = 0;
    kDistribution = 0;
    startPower = 25;
    stopPower = 25;
    startK = 101;
    jumpK = 1;
    stopK = 101;
    break;
  case 3:
    distributionType = 0;
    kDistribution = 1;
    startPower = 15;
    stopPower = 15;
    startK = 101;
    jumpK = 1;
    stopK = 101;
    break;
  case 4:
    break;
  default:
    printf ("Invalid input\n");
    return 0;
  }
  // printf ("opening file\n");
  // printf ("%s\n", typeString);
  // printf ("%s\n", getDistributionOptions(type, distributionType));
  // printf ("%s\n", getKDistributionOptions(kDistribution));
  // printf ("%d\n", startPower);
  // printf ("%d\n", stopPower);
  // printf ("%d\n", startK);
  // printf ("%d\n", jumpK);
  // printf ("%d\n", stopK);
  // printf ("%d\n", testCount);
  // printf ("%s\n", hostName);
  // printf ("%s\n", humanTime);
  
  snprintf(fileName, 128, 
           "%s %s k-dist:%s 2^%d to 2^%d (%d:%d:%d) %d-tests on %s at %s", 
           typeString, getDistributionOptions(type, distributionType), 
           getKDistributionOptions(kDistribution), startPower, stopPower, 
           startK, jumpK, stopK, testCount, hostName, humanTime);
  printf("File Name: %s \n", fileName);
  
  printf ("running\n");
  switch(type){
  case 0:
    runTests<float>(distributionType,fileName,startPower,stopPower,testCount,
                    kDistribution,startK,stopK,jumpK);
    break;
  case 1:
    runTests<double>(distributionType,fileName,startPower,stopPower,testCount,
                     kDistribution,startK,stopK,jumpK);
    break;
  case 2:
    runTests<uint>(distributionType,fileName,startPower,stopPower,testCount,
                   kDistribution,startK,stopK,jumpK);
    break;
  default:
    printf("You entered and invalid option, now exiting\n");
    break;
  }

  free (fileName);
  return 0;
}

