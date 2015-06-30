SHELL = /bin/sh
CUDA_INSTALL_PATH ?= /usr/local/cuda

CPP := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc #-ccbin /usr/bin
.SUFFIXES: .c .cpp .cu .o

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I./lib/ -I./moderngpu/include -I/usr/include/openmpi
# Libraries
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcurand -lcudart
# LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcurand -lm -lgsl -lgslcblas
LIB_MPI := -L/usr/lib/openmpi -lmpi
# ARCH
ARCH = -arch=sm_20
# dynamic parallelism
CDP = -DCUB_CDP

# Common flags
 COMMONFLAGS += $(INCLUDES)
 COMMONFLAGS += -g
# Compilers
NVCCFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(ARCH)
NVCCFLAGS += $(LIB_CUDA)
NVCCFLAGS += $(LIB_MPI)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

#NVCCFLAGS += $(CDP)

PROGRAMS = \
compareMultiselect \
analyzeMultiselect \
realDataTests \
compareMultiselectNew2 \
compareMultiselectNewFindK

SMOS = \
SMOStimingsOSDistrAll \
SMOStimingsOSDistrUniform \
SMOStimingsVectorGrowth \
SMOStimingsTableData \
SMOSanalyze \
realDataTests \
SMOStimingsCPUvGPU \
ProcessData \
CPUProcessData

CompareMultiselect = \
compareMultiselect.cu \
bucketMultiselect.cu naiveBucketMultiselect.cu \
bucketMultiselect_thrust.cu naiveBucketMultiselect.cu \
generateProblems.cu multiselectTimingFunctions.cu

CompareMultiselectNew2 = \
compareMultiselectNew2.cu \
bucketMultiselectNew2.cu naiveBucketMultiselect.cu \
bucketMultiselect_thrust.cu naiveBucketMultiselect.cu \
generateProblems.cu multiselectTimingFunctionsNew2.cu

CompareMultiselectNewFindK = \
compareMultiselectNewFindK.cu \
bucketMultiselectNewFindK.cu naiveBucketMultiselect.cu \
bucketMultiselect_thrust.cu naiveBucketMultiselect.cu \
generateProblems.cu multiselectTimingFunctionsNew2.cu \
recursionKernels.cu findk.cu 

AnalyzeMultiselect = \
analyzeMultiselect.cu \
bucketMultiselectNew.cu \
multiselectTimingFunctions.cu

RealDataTests = \
realDataTests.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOStimingsOSDistrAll = \
SMOStimingsOSDistrAll.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOStimingsOSDistrUniform = \
SMOStimingsOSDistrUniform.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOStimingsVectorGrowth = \
SMOStimingsVectorGrowth.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOStimingsTableData = \
SMOStimingsTableData.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOSanalyze = \
SMOSanalyze.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOStimingsCPUvGPU = \
SMOStimingsCPUvGPU.cu \
bucketMultiselect.cu \
generateProblems.cu \
quickMultiSelect.cpp

sortcompare = \
moderngpu/src/mgpucontext.cu \
moderngpu/src/mgpuutil.cpp \
sortcompare.cu

all: $(PROGRAMS)

allSMOS: $(SMOS)

compareMultiselect_working: $(CompareMultiselect_working)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

compareMultiselect: $(CompareMultiselect)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

compareMultiselectNew2: $(CompareMultiselectNew2)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

compareMultiselectNewFindK: $(CompareMultiselectNewFindK)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

analyzeMultiselect: $(AnalyzeMultiselect)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

realDataTests: $(RealDataTests)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

SMOStimingsOSDistrAll: $(SMOStimingsOSDistrAll)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

SMOStimingsVectorGrowth: $(SMOStimingsVectorGrowth)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

SMOStimingsOSDistrUniform: $(SMOStimingsOSDistrUniform)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

SMOStimingsTableData: $(SMOStimingsTableData)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

SMOSanalyze: $(SMOSanalyze)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

SMOStimingsCPUvGPU: $(SMOStimingsCPUvGPU)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 

ProcessData: readMultiselectOutputfile.cpp
	$(CXX) -o readMultiselectOutput readMultiselectOutputfile.cpp $(CXXFLAGS)

CPUProcessData: CPUreadMultiselectOutputfile.cpp
	$(CXX) -o CPUreadMultiselectOutput CPUreadMultiselectOutputfile.cpp $(CXXFLAGS)

sortcompare: $(sortcompare)
	$(NVCC) -o $@ $(NVCCFLAGS) moderngpu/src/mgpucontext.cu moderngpu/src/mgpuutil.cpp $(addsuffix .cu,$@) 
 
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(PROGRAMS) *~ *.o

cleanSMOS:
	rm -rf $(SMOS) *~ *.o



