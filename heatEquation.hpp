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
// #include <iostream>
// #include <fstream>
#include <string>
// #include <cerrno>
// #include <cstring>
// #include <stdexcept>

//GPU library include statements
#include <cuda.h>


struct HeatProblem1d {
	int l; //Length of the rod
	float alpha; //Thermal diffusivity
	float deltaT; //Distance between time points
	float deltaX; //distance between discrete points on rod
	float leftTemp; //Temperature at left end of rod (held constant)
	float rightTemp; //Temperature at right end of rod (held constant)

	//Pointer to function which provides starting temps
	//This function must be declared with device!
	float (*initFunction)(float position);

	//Constructor
	HeatProblem1d(int l, float alpha, float deltaT, float deltaX, float leftTemp, float rightTemp,float (*initFunction)(float)):
	l(l), alpha(alpha), deltaT(deltaT), deltaX(deltaX), leftTemp(leftTemp), rightTemp(rightTemp), initFunction(initFunction) {};
};

/**
 *Numerically sloves the given heat equation problem and returns a pointer to the
 *desired data.
 *
 *@param problemParameters A struct containing all the information needed to solve the problem.
 *@param everyXMoments The number of time intervals that will be allowed to elapse between
 succesive recordings of state.
 *@param
 *@return A pointer to the a 2d array holding the state of the system at periodic moments in
 *time. The moments will be t = 0, t = everyXMoments, t = 2 * everyXMoments, etc. If A is
 
*/
__host__ float *sloveProblemInstance(HeatProblem1d problemParameters, std::string fileName) {
	//Malloc the memory to store the results on the host
	//Invoke the kernel
	//copy back the data
}
#endif
