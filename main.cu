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
	cout << "TODO: implement main" << endl;
	HeatProblem1d(1, 1, .001, .1, 0, 0, initFunction1);
	return 0;
}
