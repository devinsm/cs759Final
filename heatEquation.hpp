/*
 *This file holds the (usually CUDA) functions which I wrote to numerically simulate
 *the diffusion of heat using the heat equation. Please see my final project report for
 *a list of resources I consulted while writing the following code.
 *
 *@author Devin McAllester
*/

#ifndef HEATEQUATION_HPP
#define HEATEQUATION_HPP
//C++ include statements
#include <iostream>
#include <cmath>
#include <ios>
#include <iomanip>

//Project specific include statements
#include "errorMacros.hpp"

//GPU library include statements
#include <cuda.h>

#define CUDA_ERROR_CHECK //turn on error checking
#define BLOCK_SIZE 512

template <typename FunctorType>
struct HeatProblem1d {
	float l; //Length of the rod
	float alpha; //Thermal diffusivity
	float leftTemp; //Temperature at left end of rod (held constant)
	float rightTemp; //Temperature at right end of rod (held constant)

	//Functor which provides starting temps. operator() should be declared with
	//__host__ and __device__. It should take a float and return a float.
	FunctorType initFunction;
};

struct SimulationParams1D {
	float deltaX; //Space between discrete points on "rod" used for calculations.
	float deltaT; //Time between iterations of finite difference calculations/updates of state.

	//Number of times finite difference calculations will be carried out (does not include t = 0).
	//For instance if numIterations == 3 then the state will be updated at t = deltaT, t = deltaT * 2,
	//t = deltaT * 3.
	int numIterations;

	//The number of time intervals that will be allowed to elapse between succesive writes
	//to output data structure.
	int periodOfRecordings;
};

/**
 *Calculates the number of moments in time for which the simulation will perform
 *the finite difference calculations.
 *
 *@param simParams The simulation parameters.
 *@return The number of moments in time for which the simulation will perform the
 *finite difference calculations (including 0 and last moment).
*/
__host__ __device__ inline int numMoments(const SimulationParams1D& simParams){
	return simParams.numIterations / simParams.periodOfRecordings + 1;
}

/**
 *Calculates the number of points on the rod that will be used in the simulation.
 *
 *@param simParams The simulation parameters.
 *@param problemParams The problem parameters.
 *@return The number of points on the rod that will be used in the simulation (includes
 *the ends of the rod).
*/
template <typename T>
__host__ __device__ inline int numPoints(const HeatProblem1d<T>& problemParams, const SimulationParams1D& simParams) {
	return ceil((problemParams.l / simParams.deltaX) + 1);
}

/**
 *Gives the index into a 2d array stored in row major order.
 *
 *@param rowIdx The index of the row.
 *@param colIdx The index of the col.
 *@param numCols The number of columns in a row.
 *@return The index in the 1d representation.
*/
__host__ __device__ inline int realIdx(int rowIdx, int colIdx, int numCols) {
	return rowIdx * numCols + colIdx;
}
/**
 *Prints the data needed for the animation to the named file.
 *
 *@param outPut The output of the simulation in row major order.
 *@param problemParams The problem parameters.
 *@param simParams The simulation parameters.
 *@param fileName The name of the file to write to.
*/
template <typename T>
__host__ void printOutPut(float *outPut, HeatProblem1d<T> problemParams,
														SimulationParams1D simParams, std::string fileName) {
	int numberOfXPoints = numPoints(problemParams, simParams);
	int numberOfMoments = numMoments(simParams);

	std::cout << simParams.deltaT * simParams.periodOfRecordings << std::endl;

	//fixed point, and 4 decimal precision, and right justifaction make output more
	//readable for humans
	std::cout << std::fixed << std::setprecision(4) << std::right;
	for (size_t i = 0; i < numberOfXPoints - 1; i++) {
		std::cout << std::setw(10) << i * simParams.deltaX << ",";
	}
	std::cout << std::setw(10) << problemParams.l << std::endl;

	for (size_t i = 0; i < numberOfMoments; i++) {
		for (size_t j = 0; j < numberOfXPoints - 1; j++) {
			std::cout << std::setw(10) << outPut[realIdx(i, j, numberOfXPoints)] << ",";
		}
		std::cout << std::setw(10) <<
				outPut[realIdx(i, numberOfXPoints - 1, numberOfXPoints)] << std::endl;
	}
	std::cout << std::setprecision(6) << std::defaultfloat; //revert to defaults
}

/**
 *The kernel which is called to actually carry out the simulation. See sloveProblemInstance
 *for details of what is stored in outPut.
 *
 *@param problemParams A struct which describes the problem being solved.
 *@param simParams Parameters of the simulation.
 *@param workingMem A 2D array in which threads can store the state at every iteration.
 *It should have the same number of columns as outPut, but only 2 rows.
 *@param output A pointer to the global memory where the result will be stored.
 *@param numCols The number of columns in the output array.
*/
template <typename T>
__global__ void sloveProblemInstanceDevice(HeatProblem1d<T> problemParams, SimulationParams1D simParams,
																					float *outPut, float* workingMem) {
	int numCols = numPoints(problemParams, simParams);
	int column = blockDim.x * blockIdx.x + threadIdx.x;
	if (column >= numCols) {
		return;
	}
	const float r = (problemParams.alpha * simParams.deltaT) / (simParams.deltaX * simParams.deltaX);

	//not valid if column == numCols - 1
	float point = column * simParams.deltaX;

	if (column == 0) {
		workingMem[realIdx(0, 0, numCols)] = problemParams.leftTemp;
	} else if (column == numCols - 1) {
		workingMem[realIdx(0, column, numCols)] = problemParams.rightTemp;
	} else {
		workingMem[realIdx(0, column, numCols)] = problemParams.initFunction(point);
	}
	outPut[realIdx(0, column, numCols)] = workingMem[realIdx(0, column, numCols)];
	__syncthreads();

	int j = 1;
	int lastTime = 0;
	int thisTime = 1;
	for (int i = 1; i <= simParams.numIterations; i++) {
		if (column == 0) {
			workingMem[realIdx(thisTime, 0, numCols)] = problemParams.leftTemp;
		} else if (column == numCols - 1) {
			workingMem[realIdx(thisTime, column, numCols)] = problemParams.rightTemp;
		} else {
			float *addrLastTime = &workingMem[realIdx(lastTime, column, numCols)];
			float newVal = (1 - 2 * r) * *addrLastTime + r * *(addrLastTime - 1) + r * *(addrLastTime + 1);

			workingMem[realIdx(thisTime, column, numCols)] = newVal;
		}

		if (i % simParams.periodOfRecordings == 0) {
			outPut[realIdx(j, column, numCols)] = workingMem[realIdx(thisTime, column, numCols)];
			j++;
		}

		lastTime = 1 - lastTime;
		thisTime = 1 - thisTime;
		__syncthreads();
	}
}

/**
 *Allocates memory on the host (by calling new) and on the device (by calling cudaMalloc)
 *to store the output of the simulation.
 *
 *@param devicePointer A reference to a pointer to store the address of the device memory.
 *@param hostPointer A reference to a pointer to store the address of the host memory.
 *@param numFloats The number of floats which will be output by the simulation.
*/
__host__ void allocateOutPutMem(float * &devicePointer, float * &hostPointer, int numFloats) {
	CudaSafeCall(cudaMalloc(&devicePointer, sizeof(float) * numFloats));
	hostPointer = new float[numFloats];
}
/**
 *Numerically sloves the given heat equation problem and returns a pointer to the
 *desired data.
 *
 *@param problemParameters A struct which describes the problem to be solved.
 *@param simParams A struct which describes the parameters of the FDM.
 *@param fileName Name of the file to which output of simulation will be printed.
 *
 *@return A pointer to the a 2d array holding the state of the system at periodic moments in
 *time. The array will be in row major order and allocated with new. The moments will be t = 0,
 *t = periodOfRecordings * deltaT, t = 2 * periodOfRecordings * deltaT, etc. If A is the returned array, then A[n][j] is
 *the temperature at the jth position along the "rod" at the time n * everyXMoments * deltaT.
 *In other words elements of rows are recordings at a moment in time.
*/
template <typename T>
__host__ float *sloveProblemInstance(HeatProblem1d<T> problemParams,
																		SimulationParams1D simParams, std::string fileName) {
	//Malloc the memory to store the results on the host

	const int numberOfXPoints = numPoints(problemParams, simParams);
	const int numberOfMoments = numMoments(simParams);

	const int sizeOfOutPutArray = numberOfMoments * numberOfXPoints;

	float *deviceOutPut = nullptr;
	float *hostOutPut = nullptr;
	allocateOutPutMem(deviceOutPut, hostOutPut, sizeOfOutPutArray);

	//Invoke the kernel
	float *workingMem = nullptr;
	CudaSafeCall(cudaMalloc(&workingMem, numberOfXPoints * 2 * sizeof(float)));

	int numBlocks = (BLOCK_SIZE + numberOfXPoints - 1) / numberOfXPoints;
	sloveProblemInstanceDevice<<<numBlocks, BLOCK_SIZE>>>(problemParams, simParams, deviceOutPut, workingMem);

	//copy back the data
	CudaSafeCall(cudaMemcpy(hostOutPut, deviceOutPut, sizeOfOutPutArray * sizeof(float), cudaMemcpyDeviceToHost));

	printOutPut(hostOutPut, problemParams, simParams, fileName);

	return hostOutPut;
}
#endif
