/*
 *This file holds the functions which I wrote to numerically simulate
 *the diffusion of heat using the heat equation. Please see my final project report for
 *a list of resources I consulted while writing the following code.
 *
 *@author Devin McAllester
*/

#ifndef HEATEQUATION_HPP
#define HEATEQUATION_HPP
//C++ include statements
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <ios>
#include <iomanip>
#include <stdexcept>

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
	unsigned numIterations;

	//The number of time intervals that will be allowed to elapse between succesive writes
	//to output data structure.
	unsigned periodOfRecordings;
};

/**
 *Calculates the number of moments in time for which the simulation will write
 *the temperatures to the output array.
 *
 *@param simParams The simulation parameters.
 *@return The number of moments in time for which the simulation will write the
 *temperatures to the output array.
*/
__host__ __device__ inline unsigned numMoments(const SimulationParams1D& simParams){
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
__host__ __device__ inline unsigned numPoints(const HeatProblem1d<T>& problemParams, const SimulationParams1D& simParams) {
	return ceil((problemParams.l / simParams.deltaX) + 1);
}

/*Calculates r given simulation params.
 *
 *@param problemParams The problem parameters.
 *@param simParams The simulation params.
 *@return (alpha * deltaT) / (deltaX ^ 2)
*/
template <typename T>
__host__ __device__ inline float rVal(const HeatProblem1d<T>& problemParams, const SimulationParams1D& simParams) {
	return (problemParams.alpha * simParams.deltaT) / (simParams.deltaX * simParams.deltaX);
}

/**
 *Gives the index into a 2d array stored in row major order.
 *
 *@param rowIdx The index of the row.
 *@param colIdx The index of the col.
 *@param numCols The number of columns in a row.
 *@return The index in the 1d representation.
*/
__host__ __device__ inline unsigned realIdx(unsigned rowIdx, unsigned colIdx, unsigned numCols) {
	return rowIdx * numCols + colIdx;
}

/**
 *Opens the named file and throws an exception if it couldn't be opened.
 *
 *@param fileName The name of the file to open.
 *@return An ofstream which can be used to write to the named file.
 *@throw std::runtime_error If the file can't be opened.
*/
std::ofstream openFile(const std::string& fileName) {
	std::ofstream outFile(fileName);

	if (!outFile.is_open()) {
		throw std::runtime_error("Failed to open file " + fileName + " (" + strerror(errno) + ")");
	}

	return outFile;
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
__host__ void printOutPut(float *outPut, const HeatProblem1d<T>& problemParams,
														const SimulationParams1D& simParams, std::string fileName) {
	unsigned numberOfXPoints = numPoints(problemParams, simParams);
	unsigned numberOfMoments = numMoments(simParams);
	std::ofstream outFile = openFile(fileName);

	outFile << simParams.deltaT * simParams.periodOfRecordings << std::endl;

	//fixed point, and 4 decimal precision, and right justifaction make output more
	//readable for humans
	outFile << std::fixed << std::setprecision(4) << std::right;
	for (size_t i = 0; i < numberOfXPoints - 1; i++) {
		outFile << std::setw(10) << i * simParams.deltaX << ",";
	}
	outFile << std::setw(10) << problemParams.l << std::endl;

	for (size_t i = 0; i < numberOfMoments; i++) {
		for (size_t j = 0; j < numberOfXPoints - 1; j++) {
			outFile << std::setw(10) << outPut[realIdx(i, j, numberOfXPoints)] << ",";
		}
		outFile << std::setw(10) <<
				outPut[realIdx(i, numberOfXPoints - 1, numberOfXPoints)] << std::endl;
	}

	outFile.close();
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
*/
template <typename T>
__global__ void sloveProblemInstanceDevice(HeatProblem1d<T> problemParams, SimulationParams1D simParams,
																					float *outPut, float* workingMem) {
	unsigned numCols = numPoints(problemParams, simParams);
	unsigned column = blockDim.x * blockIdx.x + threadIdx.x;
	if (column >= numCols) {
		return;
	}
	const float r = rVal(problemParams, simParams);

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

	int outPutRow = 1;
	int lastTime = 0;
	int thisTime = 1;
	for (unsigned i = 1; i <= simParams.numIterations; i++) {
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
			outPut[realIdx(outPutRow, column, numCols)] = workingMem[realIdx(thisTime, column, numCols)];
			outPutRow++;
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
__host__ void allocateOutPutMem(float * &devicePointer, float * &hostPointer, unsigned numFloats) {
	CudaSafeCall(cudaMalloc(&devicePointer, sizeof(float) * numFloats));
	hostPointer = new float[numFloats];
}
/**
 *Numerically sloves the given heat equation problem. It prints the output to
 *the named file. Please see my project report for a detailed description of
 *the data produced by the simulation, as well as how to run the animation script.
 *
 *@param problemParameters A struct which describes the problem to be solved.
 *@param simParams A struct which describes the parameters used to carry out the simulation.
 *@param fileName Name of the file to which output of simulation will be printed.
 *
 *@throws std::runtime_error If the file can't be opened or if the simulation
 *parameters would make the simulation numerically unstable (i.e. r > 1/2).
*/
template <typename T>
__host__ float *sloveProblemInstance(HeatProblem1d<T> problemParams,
																		SimulationParams1D simParams, std::string fileName) {
	if (rVal(problemParams, simParams) > .5) {
		throw std::runtime_error("r for simulation params is greater than 1/2");
	}
	//Malloc the memory to store the results on the host

	const unsigned numberOfXPoints = numPoints(problemParams, simParams);
	const unsigned numberOfMoments = numMoments(simParams);

	const unsigned sizeOfOutPutArray = numberOfMoments * numberOfXPoints;

	float *deviceOutPut = nullptr;
	float *hostOutPut = nullptr;
	allocateOutPutMem(deviceOutPut, hostOutPut, sizeOfOutPutArray);

	//Invoke the kernel
	float *workingMem = nullptr;
	CudaSafeCall(cudaMalloc(&workingMem, numberOfXPoints * 2 * sizeof(float)));

	unsigned numBlocks = (BLOCK_SIZE + numberOfXPoints - 1) / numberOfXPoints;
	sloveProblemInstanceDevice<<<numBlocks, BLOCK_SIZE>>>(problemParams, simParams, deviceOutPut, workingMem);

	//copy back the data
	CudaSafeCall(cudaMemcpy(hostOutPut, deviceOutPut, sizeOfOutPutArray * sizeof(float), cudaMemcpyDeviceToHost));

	printOutPut(hostOutPut, problemParams, simParams, fileName);

	delete[] hostOutPut;
	CudaSafeCall(cudaFree(deviceOutPut));
	return hostOutPut;
}
#endif
