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
// #include <fstream>
#include <string>
// #include <cerrno>
// #include <cstring>
// #include <stdexcept>

//GPU library include statements
#include <cuda.h>


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
 *Allocates memory on the host (by calling new) and on the device (by calling cudaMalloc)
 *to store the output of
*/

/**
 *Numerically sloves the given heat equation problem and returns a pointer to the
 *desired data.
 *
 *@param problemParameters A struct wich describes the problem to be solved.
 *@param simulationParams A struct which describes the parameters of the FDM.
 *
 *@return A pointer to the a 2d array holding the state of the system at periodic moments in
 *time. The array will be in row major order and allocated with new. The moments will be t = 0,
 *t = everyXMoments * deltaT, t = 2 * everyXMoments * deltaT, etc. If A is the returned array, then A[n][j] is
 *the temperature at the jth position along the "rod" at the time n * everyXMoments * deltaT.
 *In other words elements of rows are recordings at a moment in time.
*/
__host__ float *sloveProblemInstance(HeatProblem1d problemParams, SimulationParams1D simParams) {
	//Malloc the memory to store the results on the host

	//number of discrete points (we always include the ends of the rod)
	const int numberOfXPoints = ceil((problemParams.l / simParams.deltaX) + 1);

	//number of moments in time (including 0 and last moment)
	const int numberOfMoments = simParams.numIterations + 1;

	const int sizeOfOutPutMem = sizeof(float) * numberOfMoments * numberOfXPoints;

	std::cout << "Number of position points: " << numberOfXPoints << std::endl;
	std::cout << "Number of time points: " << numberOfMoments << std::endl;
	std::cout << "Size of output memory: " << sizeOfOutPutMem << std::endl;

	// float *deviceOutPut, *hostOutPut;
	// allocateOutPutMem(deviceOutPut, hostOutPut);

	//Invoke the kernel
	//copy back the data

	return new float;
}
#endif
