//
//  quickSelect.c
//
//  Created by Patrick Slough on 11/15/14.
//
// gcc -std=c99 -o qs quickSelect.c
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//#define VERBOSE


template<typename T>
void swap(T * x, T * y){
 	//generic pointer swap
    T temp = *x;
    *x = *y;
    *y = temp;
}



template<typename T>
uint* mypartition(T * vec, uint length, T pivot){
    
	// Dutch National flag partition algorithm
	// inspired by algorithm discussed here:
	// http://www.math.grin.edu/~rebelsky/Courses/CSC207/2014F/eboards/eboard.24.html

    // index of where elements smaller than the pivot end
    uint s = 0;
    //index of where equal elements end
    uint e = 0;
    //index of where bigger elements start
    uint b = length;
    //make array to return s and b
    uint* returnVals;
    returnVals = (uint *)malloc(sizeof(uint)*2);
	
    while (e < b){
    	if (vec[e] < pivot){
        	swap<T>(&vec[s++], &vec[e++]);
        }
        else if (vec[e] == pivot){
            e++;
        }
        else{
            swap<T>(&vec[e], &vec[b - 1]);
            b--;
        }
    }
	returnVals[0] = s;
	returnVals[1] = b;
	return returnVals;
}



template<typename T>
T myqs(T * vec, uint orderStat, uint length){
    if (length == 1){
      return vec[0];
    } else {
	uint mid = length/2;
	T pivot = vec[mid];
	
	uint *bounds = mypartition<T>(vec, length, pivot);

	if(orderStat >= bounds[0] && orderStat < bounds[1]){
		return vec[orderStat];
	}
	else if(orderStat < bounds[0]){
		return myqs<T>(vec, orderStat, bounds[0]);
	}
	else{
		return myqs<T>(vec+bounds[1], orderStat-bounds[1], length-bounds[1]);
	}
    }
}


template<typename T>
T myquickSelect(T* vec, uint orderStat, uint length){
	if(orderStat > length){
		return -1;
	}
	else{
		return myqs<T>(vec, orderStat, length);
	}
}


template<typename T>
T* naiveQuickMultiSelect(T* vec, uint* orderStats, uint vecLength, uint orderStatsLength){
	
	//vector to hold returned orderStats
	T* returnVals = malloc(sizeof(T)*orderStatsLength);
	
	//run quickselect on every orderStat wanted. 
	//vec should become more sorted on every run, making selection faster
	for(int i = 0; i < orderStatsLength; i++){
		returnVals[i] = myquickSelect<T>(vec, orderStats[i], vecLength);
	}
	
	return returnVals;
		
}


template<typename T>
void myqms(T * vec, uint * orderStats, T * results, uint vecLength, uint osLength){

	if(osLength <= 1){
		results[0] = myqs<T>(vec, orderStats[0], vecLength);
                //printf("result = %d \n", results[0]);  
	} else {
		uint mid = (osLength)/2;
                // *** median of threes ***
                if (vecLength>10){
                  if (vec[0] > vec[vecLength-1]){
        	    swap<T>(&vec[0], &vec[vecLength-1]);
                  }
                  if (vec[0] > vec[mid]){
        	    swap<T>(&vec[0], &vec[mid]);
                  }
                  if (vec[vecLength-1] < vec[mid]){
        	    swap<T>(&vec[0], &vec[mid]);
                  }
                }


		uint w = orderStats[mid];
		results[mid] = myqs<T>(vec, w, vecLength);
		//printf("w = %d, result[%d] = %d\n", w, mid, results[mid]);
            // if there are order statistcs smaller than w, recurse
            if (mid>0) {
		myqms<T>(vec, orderStats, results, w, mid);
            }

            // if there are oder statistics larger than w, recurse   
            if (osLength > mid + 1){
                // update the order statistics for the new vector
                for (int i=mid+1; i<osLength; i++){
                  orderStats[i] -= (w+1);
                }
		myqms<T>(vec + w + 1, orderStats + mid + 1, results + mid + 1, vecLength - w - 1, osLength - mid -1);
            } // end if osLength>mid+1

	}  // end if {osLength<=1} else {
}

/*
template<typename T>
void quickMultiSelect(T * vec, uint * orderStats, T * results, uint vecLength, uint orderStatsLength){
	//based on mselect algorithm from 
	//http://www.ccse.kfupm.edu.sa/~suwaiyel/publications/multiselection_parCom.pdf
	
    uint *osIndices;
    osIndices = (uint *) malloc(sizeof(uint)*orderStatsLength);
    for (register int i=0; i< orderStatsLength; i++){
      osIndices[i]=i;
    }
    // sort the given indices
    thrust::host_ptr<uint>kVals_ptr(orderStats);
    thrust::host_ptr<uint>kIndices_ptr(osIndices);
    thrust::sort_by_key(kVals_ptr, kVals_ptr + numKs, kIndices_ptr);
	
    myqms<T>(vec, orderStats, results, vecLength, orderStatsLength);
	
    free(osIndices);
}
*/

  /* Wrapper function around the multi-selection fucntion that inverts the given k indices.
   */
  template <typename T>
  T quickMultiselectWrapper (T * h_vector, uint length, uint * kVals_ori, uint numKs
                              , T * outputs, int blocks, int threads) { 

    uint kVals[numKs];

    // fix zero indexing
    for (register int i = 0; i < numKs; i++) 
      kVals[numKs - i - 1] = length - kVals_ori[i];
		
//    quickMultiSelect<T>(h_vector, kVals, outputs, length, numKs);
  myqms<T>(h_vector, kVals, outputs, length, numKs);

        for (register int i=0; i<numKs/2; i++)
          swap<T>(&outputs[i],&outputs[numKs-i-1]);
   
    //outputs = quickMultiSelect<T>(h_vector, length, kVals, numKs);

    return 1;
  }



