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


struct HeatProblem1d {
	float l; //Length of the rod
	float alpha; //Thermal diffusivity
	float leftTemp; //Temperature at left end of rod (held constant)
	float rightTemp; //Temperature at right end of rod (held constant)

	//Pointer to function which provides starting temps
	//This function must be declared with device!
	float (*initFunction)(float position);
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
 *Prints a 2D array in a table.
 *
 *@param array A pointer to the array (which should be in row major order).
 *@param numCols The number of columns.
 *@param numRows The number of rows.
*/
__host__ void print2dArray(float *array, int numCols, int numRows) {
	std::cout << std::fixed << std::setprecision(4) << std::right;
	for (size_t i = 0; i < numRows; i++) {
		for (size_t j = 0; j < numCols; j++) {
			std::cout << std::setw(7) << array[realIdx(i, j, numCols)];
		}
		std::cout << std::endl;
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
__global__ void sloveProblemInstanceDevice(HeatProblem1d problemParams, SimulationParams1D simParams,
																					float *outPut, float* workingMem, int numCols) {
	int column = blockDim.x * blockIdx.x + threadIdx.x;
	if (column >= numCols) {
		return;
	}

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
			workingMem[realIdx(thisTime, column, numCols)] = problemParams.initFunction(point);
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
 *
 *@return A pointer to the a 2d array holding the state of the system at periodic moments in
 *time. The array will be in row major order and allocated with new. The moments will be t = 0,
 *t = periodOfRecordings * deltaT, t = 2 * periodOfRecordings * deltaT, etc. If A is the returned array, then A[n][j] is
 *the temperature at the jth position along the "rod" at the time n * everyXMoments * deltaT.
 *In other words elements of rows are recordings at a moment in time.
*/
__host__ float *sloveProblemInstance(HeatProblem1d problemParams, SimulationParams1D simParams) {
	//Malloc the memory to store the results on the host

	//number of discrete points (we always include the ends of the rod)
	const int numberOfXPoints = ceil((problemParams.l / simParams.deltaX) + 1);

	//number of moments in time (including 0 and last moment)
	const int numberOfMoments = simParams.numIterations / simParams.periodOfRecordings + 1;

	const int sizeOfOutPutArray = numberOfMoments * numberOfXPoints;

	std::cout << "Number of position points: " << numberOfXPoints << std::endl;
	std::cout << "Number of time points recorded: " << numberOfMoments << std::endl;
	std::cout << "Size of output memory: " << sizeOfOutPutArray << std::endl;

	float *deviceOutPut = nullptr;
	float *hostOutPut = nullptr;
	allocateOutPutMem(deviceOutPut, hostOutPut, sizeOfOutPutArray);

	//Invoke the kernel
	float *workingMem = nullptr;
	CudaSafeCall(cudaMalloc(&workingMem, numberOfXPoints * 2 * sizeof(float)));

	int numBlocks = (BLOCK_SIZE + numberOfXPoints - 1) / numberOfXPoints;
	sloveProblemInstanceDevice<<<numBlocks, BLOCK_SIZE>>>(problemParams, simParams, deviceOutPut, workingMem, numberOfXPoints);

	//copy back the data
	cudaMemcpy(hostOutPut, deviceOutPut, sizeOfOutPutArray * sizeof(float), cudaMemcpyDeviceToHost);

	print2dArray(hostOutPut, numberOfXPoints, numberOfMoments);

	return new float;
}
#endif
