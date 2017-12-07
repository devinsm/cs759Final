/*
 *This file contains the main program which calls the functions in heatEquation.hpp.
 *
 *@author Devin McAllester
*/

//C++ include statements
#include <iostream>
#include <exception>

//Project specific include statements
#include "heatEquation.hpp"

//Using statements for convience
using std::cout;
using std::endl;

struct TestFunctor {
	int i = 1;
	__host__ __device__
	float operator() (float position) {
		return position * i++;
	};
};

struct SmithFunctor {
	__host__ __device__
	float operator() (float position) {
		if (position <= .5) {
			return 2 * position;
		} else {
			return 2 * (1 - position);
		}
	};
};

int main(int argc, char const *argv[]) {
	HeatProblem1d<SmithFunctor> problem1;
	problem1.l = 1;
	problem1.alpha = 1;
	problem1.leftTemp = 0;
	problem1.rightTemp = 0;

	SimulationParams1D simParams1;
	simParams1.deltaX = .1;
	simParams1.deltaT = .001;
	simParams1.numIterations = 500;
	simParams1.periodOfRecordings = 1;

	try {
		sloveProblemInstance(problem1, simParams1, "smithProblem.txt");
	} catch (std::exception& e) {
		cout << "Simulation on line " << __LINE__ - 2 << "threw an exception" << endl;
		cout << e.what() << endl;
	}
	return 0;
}
