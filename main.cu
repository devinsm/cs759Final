/*
 *This file contains the main program which calls the functions in heatEquation.hpp.
 *For now it writes the output data to plain text file. Once I get the graphics/animation
 *part of the program nailed down I might change this code to match.
*/

//C++ include statements
#include <iostream>

//GPU library include statements

//Project specific include statements
#include "heatEquation.hpp"

//Using statements for convience
using std::cout;
using std::endl;

__device__ float initFunction1(float position) {
	return .1f;
}

int main(int argc, char const *argv[]) {
	HeatProblem1d problem1;
	problem1.l = 1;
	problem1.alpha = 1;
	problem1.leftTemp = 0;
	problem1.rightTemp = 0;
	problem1.initFunction = initFunction1;

	SimulationParams1D simParams1;
	simParams1.deltaX = .15;
	simParams1.deltaT = .001;
	simParams1.numIterations = 20;
	simParams1.periodOfRecordings = 1;

	sloveProblemInstance(problem1, simParams1);
	return 0;
}
